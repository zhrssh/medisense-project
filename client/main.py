import cv2
import requests
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medisense")
        self.root.geometry("420x400")

        self.camera = None
        self.captured_image_path = None
        self.canvas = None

        # GUI components
        self.label = tk.Label(self.root, text="Medisense", font=("Helvetica", 16))
        self.label.grid(row=0, column=0, columnspan=2, pady=10)

        self.label = tk.Label(self.root, text="Menu", font=("Helvetica", 14))
        self.label.grid(row=1, column=0, columnspan=1, pady=10)

        self.label = tk.Label(self.root, text="Camera", font=("Helvetica", 14))
        self.label.grid(row=1, column=1, columnspan=1, pady=10)

        self.button_frame = tk.Frame(self.root)
        self.button_frame.grid(row=2, column=0, padx=10, pady=10)

        self.camera_frame = tk.Frame(self.root)
        self.camera_frame.grid(row=2, column=1)

        # Add a Label to display the captured image
        self.captured_image_label = tk.Label(self.root)
        self.captured_image_label.grid(row=3, column=1, pady=10)

        self.start_button = tk.Button(
            self.button_frame, text="Start", command=self.start_camera
        )
        self.start_button.grid(row=0, column=0, pady=5)

        self.stop_button = tk.Button(
            self.button_frame, text="Stop", command=self.stop_camera, state=tk.DISABLED
        )
        self.stop_button.grid(row=1, column=0, pady=5)

        self.capture_button = tk.Button(
            self.button_frame,
            text="Capture",
            command=self.capture_image,
            state=tk.DISABLED,
        )
        self.capture_button.grid(row=2, column=0, pady=5)

        self.send_button = tk.Button(
            self.button_frame, text="Send", command=self.send_image, state=tk.DISABLED
        )
        self.send_button.grid(row=3, column=0, pady=5)

        # Start the Tkinter main loop
        self.start_camera()
        root.mainloop()

    def start_camera(self):
        if self.camera is None or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)

        if self.canvas:
            self.canvas.destroy()

        self.canvas = tk.Canvas(self.camera_frame, width=256, height=256)
        self.canvas.pack(pady=10)

        self.capture_button.config(state=tk.NORMAL)
        self.send_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)
        self.start_button.config(state=tk.DISABLED)
        self.show_camera_feed()

    def stop_camera(self):
        self.capture_button.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)

        # Release the camera capture
        if self.camera is not None and self.camera.isOpened():
            self.camera.release()

        # Destroy the canvas
        if self.canvas:
            self.canvas.destroy()

    def show_camera_feed(self):
        if self.camera is not None and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                frame = cv2.resize(frame, (256, 256))  # Resize the frame

                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                self.root.after(10, self.show_camera_feed)

    def capture_image(self):
        if self.camera is not None and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.captured_image_path = "captured_image.jpg"
                cv2.imwrite(
                    self.captured_image_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

                # Display the captured image in the Label widget
                self.view_captured_image()
        else:
            messagebox.showwarning("Capture", "Camera not available.")

    def send_image(self):
        if self.captured_image_path:
            url = "http://localhost:5000/predict"  # Replace with your server endpoint
            files = {"file": open(self.captured_image_path, "rb")}

            try:
                response = requests.post(url, files=files)
                if response.status_code == 200:
                    messagebox.showinfo(
                        "Send", f"Image '{self.captured_image_path}' sent successfully!"
                    )
                else:
                    messagebox.showerror(
                        "Error",
                        f"Failed to send image. Status code: {response.status_code}",
                    )
            except requests.RequestException as e:
                messagebox.showerror("Error", f"Request failed: {str(e)}")
        else:
            messagebox.showwarning("Send", "No image captured yet.")

    def view_captured_image(self):
        if self.camera is not None and self.camera.isOpened():
            self.stop_camera()

        if self.captured_image_path:
            # Load the captured image using OpenCV (BGR format)
            captured_image_cv2 = cv2.imread(self.captured_image_path)
            # Convert the image from BGR to RGB
            captured_image_cv2_rgb = cv2.cvtColor(captured_image_cv2, cv2.COLOR_BGR2RGB)
            # Convert the image for PIL
            captured_image_pil = Image.fromarray(captured_image_cv2_rgb)
            # Resize the image to fit within the Label
            captured_image_pil.thumbnail((256, 256))
            # Convert the image for Tkinter
            self.photo_captured = ImageTk.PhotoImage(captured_image_pil)

            if self.canvas:
                self.canvas.destroy()

            self.canvas = tk.Canvas(self.camera_frame, width=256, height=256)
            self.canvas.pack(pady=10)

            # Create an image item on the Canvas to display the captured image
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_captured)
        else:
            messagebox.showwarning("View Image", "No image captured yet.")

        self.send_button.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
