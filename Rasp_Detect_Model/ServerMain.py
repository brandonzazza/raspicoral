import argparse
import cv2
import os
import socket
from flask import Flask, jsonify
from threading import Thread, Lock
import time
import numpy as np
from collections import deque
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

# People tracking variables
track_id_counter = 0
trackers = {}  # ID: deque of centroids
max_track_length = 10
entrances = 0
exits = 0
entrances_lock = Lock()

def generate_frames(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
        time.sleep(0.05)

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def track_people(cv2_im, inference_size, objs):
    global track_id_counter, trackers, entrances, exits
    height, width, _ = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]

    center_line = height // 2
    new_centroids = []

    # Scale bounding boxes and collect centroids
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)
        center_x = (x0 + x1) // 2
        center_y = (y0 + y1) // 2
        new_centroids.append(((x0, y0, x1, y1), (center_x, center_y)))

    # Match centroids to existing trackers
    updated_trackers = {}
    used_ids = set()
    for bbox, centroid in new_centroids:
        matched = False
        for tid, points in trackers.items():
            if len(points) == 0:
                continue
            if euclidean_distance(centroid, points[-1]) < 50:  # Distance threshold
                updated_trackers[tid] = points
                updated_trackers[tid].append(centroid)
                used_ids.add(tid)
                matched = True
                break

        if not matched:
            # New person
            updated_trackers[track_id_counter] = deque(maxlen=max_track_length)
            updated_trackers[track_id_counter].append(centroid)
            used_ids.add(track_id_counter)
            track_id_counter += 1

    # Count entrances/exits
    for tid, points in updated_trackers.items():
        if len(points) >= 2:
            y_previous = points[-2][1]
            y_current = points[-1][1]
            if y_previous < center_line and y_current >= center_line:
                with entrances_lock:
                    entrances += 1
            elif y_previous > center_line and y_current <= center_line:
                with entrances_lock:
                    exits += 1

    # Update trackers
    trackers = updated_trackers

    # Draw detections
    for tid, points in trackers.items():
        if len(points) > 0:
            cx, cy = points[-1]
            cv2.circle(cv2_im, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(cv2_im, f'ID {tid}', (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw center line
    cv2.line(cv2_im, (0, center_line), (width, center_line), (0, 0, 255), 2)

    return cv2_im

def handle_client(client_socket, inference_size, interpreter, labels, args):
    global entrances, exits
    try:
        cap = cv2.VideoCapture(args.camera_idx)
        frame_generator = generate_frames(cap)
        while True:
            frame = next(frame_generator)
            if frame is None:
                break
            cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
            run_inference(interpreter, cv2_im_rgb.tobytes())
            objs = get_objects(interpreter, args.threshold)
            person_objs = [obj for obj in objs if labels.get(obj.id, '') == 'person']

            frame = track_people(frame, inference_size, person_objs)

            cv2.putText(frame, f'Entrances: {entrances}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, f'Exits: {exits}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            ret, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
            if not ret:
                continue
            try:
                client_socket.sendall(jpg.tobytes())
            except (ConnectionResetError, BrokenPipeError):
                break

    except Exception as e:
        print(f"Client error: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        client_socket.close()

def run_server(inference_size, interpreter, labels, args, host='', port=8080):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server running on port {port}")
    try:
        while True:
            client_socket, client_address = server_socket.accept()
            thread = Thread(target=handle_client, args=(client_socket, inference_size, interpreter, labels, args))
            thread.start()
    finally:
        server_socket.close()

app = Flask(__name__)

@app.route('/data', methods=['GET'])
def get_data():
    with entrances_lock:
        return jsonify({
            'entries': entrances,
            'exits': exits
        })

def start_flask_api():
    app.run(host="0.0.0.0", port=4000)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./model/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
    parser.add_argument('--labels', default='./model/coco_labels.txt')
    parser.add_argument('--camera_idx', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    flask_thread = Thread(target=start_flask_api, daemon=True)
    flask_thread.start()

    run_server(inference_size, interpreter, labels, args)

if __name__ == '__main__':
    main()
