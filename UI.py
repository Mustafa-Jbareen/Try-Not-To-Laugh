import tkinter as tk
from tkinter import messagebox
import cv2
import dlib
import time
from keras.api.preprocessing.image import img_to_array
from keras.api.models import load_model
import numpy as np
import imutils
import os
import threading
from ffpyplayer.player import MediaPlayer


def play_video_with_audio(file_path, stop_event):
    cap = cv2.VideoCapture(file_path)
    player = MediaPlayer(file_path)

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break  # Exit when video ends

        audio_frame, val = player.get_frame()
        if val != 'eof' and audio_frame is not None:
            img, t = audio_frame  # Play the audio frame

        cv2.imshow("Funny Video", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            stop_event.set()  # Stop playback if 'q' is pressed
            break


def play_funny_media(stop_event):
    def play_media():
        funny_folder = "funny"
        funny_files = [f for f in os.listdir(funny_folder) if f.endswith(('.mp4'))]

        if funny_files:
            while not stop_event.is_set():  # Loop until stop_event is set
                for selected_file in funny_files:
                    file_path = os.path.join(funny_folder, selected_file)

                    if selected_file.endswith('.mp4'):
                        play_video_with_audio(file_path, stop_event)
                    else:
                        image = cv2.imread(file_path)
                        cv2.imshow("Funny Image", image)
                        cv2.waitKey(3000)  # Display image for 3 seconds
                        cv2.destroyAllWindows()

                    if stop_event.is_set():  # Check if stop_event was triggered
                        break
        else:
            print("No funny media found.")

    media_thread = threading.Thread(target=play_media)
    media_thread.start()
    return media_thread


def run_detection(player_mode, duration, include_funny):
    # Paths for the face cascade and the laugh detection model
    cascade_path = "haarcascade_frontalface_default.xml"
    model_path = "best.h5"

    # Load the face detector cascade and laugh detector CNN
    detector = cv2.CascadeClassifier(cascade_path)
    model = load_model(model_path)

    # Initialize face tracking variables
    face_trackers = {}
    laugh_counts = {}
    laugh_durations = {}
    laugh_start_times = {}
    face_id = 1
    total_laughs = 0  # Track total laughs for 1 player mode

    # Set up the game duration and camera
    game_duration = duration
    camera = cv2.VideoCapture(0)  # Using webcam by default

    stop_event = threading.Event()  # Event to stop media

    if include_funny:
        media_thread = play_funny_media(stop_event)

    start_time = time.time()

    # Main game loop
    while True:
        elapsed_time = time.time() - start_time
        remaining_time = max(0, game_duration - elapsed_time)

        if elapsed_time >= game_duration:
            break

        grabbed, frame = camera.read()
        if not grabbed:
            break

        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frameClone = frame.copy()

        # Detect faces in the frame
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(180, 180), maxSize=(640, 640))

        # Track face and laugh status
        to_remove = []
        for fid, tracker in face_trackers.items():
            tracking_quality = tracker.update(frame)
            if tracking_quality < 7:
                to_remove.append(fid)
                continue

            pos = tracker.get_position()
            x = int(pos.left())
            y = int(pos.top())
            w = int(pos.width())
            h = int(pos.height())

            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Predict laugh
            (notLaughing, Laughing) = model.predict(roi)[0]
            label = "Laughing" if Laughing > notLaughing else "Not Laughing"

            if label == "Laughing":
                if laugh_start_times[fid] is None:
                    laugh_start_times[fid] = time.time()
                    laugh_counts[fid] += 1
                    if player_mode == 1:
                        total_laughs += 1  # Increment total laughs for 1 player
                else:
                    laugh_durations[fid] += time.time() - laugh_start_times[fid]
                    laugh_start_times[fid] = time.time()
            else:
                laugh_start_times[fid] = None

            # Initialize the duration for this face ID
            if fid not in laugh_durations:
                laugh_durations[fid] = 0

            # Display based on the number of players
            if player_mode == 2:
                if fid in laugh_durations:
                    cv2.putText(frameClone, f"Player: {fid} {label} - Duration: {laugh_durations[fid]:.2f}s",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 255, 0) if label == "Laughing" else (0, 0, 255), 2)
                else:
                    cv2.putText(frameClone, f"ID: {fid} {label} - Duration: N/A",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 255, 0) if label == "Laughing" else (0, 0, 255), 2)
                cv2.rectangle(frameClone, (x, y), (x + w, y + h),
                              (0, 255, 0) if label == "Laughing" else (0, 0, 255), 2)
            else:
                cv2.putText(frameClone, f"{label}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 0) if label == "Laughing" else (0, 0, 255), 2)
                cv2.rectangle(frameClone, (x, y), (x + w, y + h),
                              (0, 255, 0) if label == "Laughing" else (0, 0, 255), 2)

        for fid in to_remove:
            del face_trackers[fid]

        # Check for new faces
        for (x, y, w, h) in faces:
            match_found = False
            for fid, tracker in face_trackers.items():
                pos = tracker.get_position()
                tx = int(pos.left())
                ty = int(pos.top())
                tw = int(pos.width())
                th = int(pos.height())
                if abs(tx - x) < 50 and abs(ty - y) < 50:
                    match_found = True
                    break

            if not match_found:
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x, y, x + w, y + h)
                tracker.start_track(frame, rect)
                face_trackers[face_id] = tracker
                laugh_counts[face_id] = 0
                laugh_durations[face_id] = 0
                laugh_start_times[face_id] = None
                face_id += 1

        # Display the remaining time
        cv2.putText(frameClone, f"Time Remaining: {int(remaining_time)}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        cv2.imshow("Face", frameClone)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # End of game logic
    stop_event.set()  # Stop the funny media playback

    if player_mode == 1:
        total_laugh_duration = sum(laugh_durations.values())
        messagebox.showinfo("Game Over", f"Total Laugh Duration: {total_laugh_duration:.2f} seconds")
    else:
        # Determine winner in 2-player mode
        if len(laugh_durations) >= 1:
            winner = min(laugh_durations, key=laugh_durations.get)
            messagebox.showinfo("Game Over",
                                f"Player {winner} wins with {laugh_durations[winner]:.2f} seconds of Laughing!")
        else:
            messagebox.showinfo("Game Over", "Not enough players detected to determine a winner!")

    # Cleanup
    camera.release()
    cv2.destroyAllWindows()


def start_detection():
    player_mode = int(player_var.get())
    duration = int(duration_var.get())
    include_funny = include_funny_var.get()  # Get the value of the funny media option
    response = messagebox.askokcancel("Start Detection",
                                      f"Starting game with {player_mode} player(s) for {duration} seconds."
                                      f"{' Including funny media!' if include_funny else ''}")
    if response:
        run_detection(player_mode, duration, include_funny)


def create_menu():
    global player_var, duration_var, include_funny_var

    # Create main window
    window = tk.Tk()
    window.title("Laugh Detection Game")
    window.geometry("500x400")  # Set the size of the window
    window.configure(bg='lightblue')  # Change background color to light blue

    # Player mode selection
    tk.Label(window, text="Select Number of Players:", bg='lightblue').pack(pady=10)
    player_var = tk.StringVar(value='1')
    tk.Radiobutton(window, text="1 Player", variable=player_var, value='1', bg='lightblue').pack()
    tk.Radiobutton(window, text="2 Players", variable=player_var, value='2', bg='lightblue').pack()

    # Game duration selection
    tk.Label(window, text="Select Game Duration (seconds):", bg='lightblue').pack(pady=10)
    duration_var = tk.StringVar(value='30')  # Default duration
    tk.Entry(window, textvariable=duration_var).pack()

    # Funny media inclusion option
    include_funny_var = tk.BooleanVar(value=False)  # Default to not include
    tk.Checkbutton(window, text="Include Funny Media", variable=include_funny_var, bg='lightblue').pack(pady=10)

    # Start button
    tk.Button(window, text="Start Game", command=start_detection).pack(pady=20)

    window.mainloop()


if __name__ == "__main__":
    create_menu()
