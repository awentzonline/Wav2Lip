import argparse
from dataclasses import dataclass, field
import os
import platform
import tempfile
import shutil
import subprocess
from typing import List

import cv2
import numpy as np
from tqdm import tqdm
import torch

from wav2lip import audio, face_detection
from .models import Wav2Lip


@dataclass
class Wav2LipInference:
    checkpoint_path: str
    pads: List[int] = (0, 10, 0, 0)
    face_det_batch_size: int = 16
    wav2lip_batch_size: int = 128
    resize_factor: int = 1
    fps: float = 25.
    crop: List[int] = (0, -1, 0, -1)
    box: List[int] = (-1, -1, -1, -1)
    rotate: bool = False
    nosmooth: bool = False
    mel_step_size: int = 16
    device: str = 'cpu'
    img_size: int = 96
    extra: field(default_factory=dict) = None # substitute for kwargs

    def face_detect(self, images):
        detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False, device=self.device
        )

        batch_size = self.face_det_batch_size

        while 1:
            predictions = []
            try:
                for i in tqdm(range(0, len(images), batch_size)):
                    predictions.extend(
                        detector.get_detections_for_batch(
                            np.array(images[i:i + batch_size])
                        )
                    )
            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError(
                        'Image too big to run face detection on GPU. '
                        'Please use the --resize_factor argument'
                    )
                batch_size //= 2
                print('Recovering from OOM error; New batch size: {}'.format(
                    batch_size
                ))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = self.pads
        for rect, image in zip(predictions, images):
            if rect is None:
                results.append([0, 0, 1, 1])
            else:
                y1 = max(0, rect[1] - pady1)
                y2 = min(image.shape[0], rect[3] + pady2)
                x1 = max(0, rect[0] - padx1)
                x2 = min(image.shape[1], rect[2] + padx2)
                results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not self.nosmooth:
            boxes = get_smoothened_boxes(boxes, T=5)
        results = [
            [image[y1: y2, x1:x2], (y1, y2, x1, x2)]
            for image, (x1, y1, x2, y2) in zip(images, boxes)
        ]

        del detector
        return results

    def datagen(self, frames, mels, is_static):
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.box[0] == -1:
            if not is_static:
                # BGR2RGB for CNN face detection
                face_det_results = self.face_detect(frames)
            else:
                face_det_results = self.face_detect([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = self.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = 0 if is_static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.img_size, self.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size // 2:] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(selfmel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(
                mel_batch,
                (len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1)
            )

            yield img_batch, mel_batch, frame_batch, coords_batch

    def _load_checkpoint(self, checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    def load_model(self):
        model = Wav2Lip()
        print("Load checkpoint from: {}".format(self.checkpoint_path))
        checkpoint = self._load_checkpoint(self.checkpoint_path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        self.model = model.to(self.device)
        return self.model.eval()

    def lipsync(self, face_filename, audio_filename, output_filename, tmpdir=None):
        if not os.path.isfile(face_filename):
            raise ValueError('--face argument must be a valid path to video/image file')

        elif face_filename.split('.')[1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(face_filename)]
            fps = self.fps
            is_static = True
        else:
            is_static = False
            video_stream = cv2.VideoCapture(face_filename)
            fps = video_stream.get(cv2.CAP_PROP_FPS)

            print('Reading video frames...')

            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                if self.resize_factor > 1:
                    frame = cv2.resize(
                        frame,
                        (frame.shape[1] // self.resize_factor,
                         frame.shape[0] // self.resize_factor)
                    )

                if self.rotate:
                    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

                y1, y2, x1, x2 = self.crop
                if x2 == -1:
                    x2 = frame.shape[1]
                if y2 == -1:
                    y2 = frame.shape[0]

                frame = frame[y1:y2, x1:x2]

                full_frames.append(frame)

        cleanup_tempdir = False
        if tmpdir is None:
            tmpdir = tempfile.mkdtemp()
            cleanup_tempdir = True
        os.makedirs(tmpdir, exist_ok=True)

        print("Number of frames available for inference: " + str(len(full_frames)))
        if not audio_filename.endswith('.wav'):
            print('Extracting raw audio...')
            temp_filename = os.path.join(tmpdir, 'temp.wav')
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(
                audio_filename, temp_filename
            )

            subprocess.call(command, shell=True)
            audio_filename = temp_filename

        wav = audio.load_wav(audio_filename, 16000)
        mel = audio.melspectrogram(wav)
        print(mel.shape)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? '
                'Add a small epsilon noise to the wav file and try again'
            )

        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1

        print("Length of mel chunks: {}".format(len(mel_chunks)))

        full_frames = full_frames[:len(mel_chunks)]

        batch_size = self.wav2lip_batch_size
        gen = self.datagen(full_frames.copy(), mel_chunks, is_static)

        temp_filename = os.path.join(tmpdir, 'result.avi')
        for i, (img_batch, mel_batch, frames, coords) in enumerate(
            tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))
        ):
            if i == 0:
                frame_h, frame_w = full_frames[0].shape[:-1]
                out = cv2.VideoWriter(temp_filename,
                                      cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

                f[y1:y2, x1:x2] = p
                out.write(f)

        out.release()

        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(
            audio_filename, temp_filename, output_filename
        )
        subprocess.call(command, shell=platform.system() != 'Windows')

        if cleanup_tempdir:
            shutil.rmtree(tmpdir)


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def find_all_source_file_pairs(face_root, audio_root):
    face_files = {}
    audio_files = {}
    for face_filename in os.listdir(face_root):
        name, ext = os.path.splitext(face_filename)
        if ext in ('.mp4', '.png', '.jpg', '.jpeg'):
            face_files[name] = os.path.join(face_root, face_filename)
    for audio_filename in os.listdir(audio_root):
        name, ext = os.path.splitext(audio_filename)
        if ext in ('.mp3', '.wav', '.m4a'):
            audio_files[name] = os.path.join(audio_root, audio_filename)

    pairs = []
    for key, face_filename in face_files.items():
        if key in audio_files:
            pairs.append((face_filename, audio_files[key]))
    return pairs


def find_one_image_many_wavs_sources(paths):
    confs = []
    for root_path in paths:
        face_filename = None
        for filename in os.listdir(root_path):
            name, ext = os.path.splitext(filename)
            if ext in ('.mp4', '.png', '.jpg'):
                face_filename = os.path.join(root_path, filename)
        if face_filename is None:
            print('No face found for', root_path)
            continue

        for filename in os.listdir(root_path):
            name, ext = os.path.splitext(filename)
            if ext in ('.mp3', '.wav', '.m4a'):
                audio_filename = os.path.join(root_path, filename)
                output_filename = os.path.join(root_path, name + '.mp4')
                confs.append((face_filename, audio_filename, output_filename))
    return confs


def main(args):
    wav2lip = Wav2LipInference(**vars(args))
    wav2lip.load_model()
    print("Model loaded")
    wav2lip.lipsync(args, args.face, args.audio, args.outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inference code to lip-sync videos in the wild using Wav2Lip models')

    parser.add_argument('--checkpoint_path', type=str,
                        help='Name of saved checkpoint to load weights from', required=True)

    parser.add_argument('--face', type=str,
                        help='Filepath of video/image that contains faces to use', required=True)
    parser.add_argument('--audio', type=str,
                        help='Filepath of video/audio file to use as raw audio source', required=True)
    parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                        default='results/result_voice.mp4')

    parser.add_argument('--static', type=bool,
                        help='If True, then use only first video frame for inference', default=False)
    parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                        default=25., required=False)

    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                        help='Padding (top, bottom, left, right). Please adjust to include chin at least')

    parser.add_argument('--face_det_batch_size', type=int,
                        help='Batch size for face detection', default=16)
    parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

    parser.add_argument('--resize_factor', default=1, type=int,
                        help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

    parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                        help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                        'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

    parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                        help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                        'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

    parser.add_argument('--rotate', default=False, action='store_true',
                        help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                        'Use if you get a flipped result, despite feeding a normal looking video')

    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')

    parser.add_argument('--nocuda', default=False, action='store_true',
                        help='Dont use CUDA even if available')

    parser.add_argument('--mel_step_size', default=16, type=int)

    args = parser.parse_args()
    args.img_size = 96
    args.device = 'cuda' if (torch.cuda.is_available() and not args.nocuda) else 'cpu'

    main(args)
