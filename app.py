import typer
import cv2
import numpy as np
import supervision as sv  # For handling annotations like bounding boxes and labels
from ultralytics import YOLO  # YOLO for object detection
import pyresearch  # Additional utilities from pyresearch

# Define the path to the weights file and load the pre-trained YOLO model
model = YOLO("last.pt")  # Replace with the appropriate path to your YOLO weights
app = typer.Typer()

def add_header_footer(frame, header_text="Ensuring Proper Workflow in Self-Driving Cars Using Computer Vision", footer_text="Pyresearch"):
    """
    Function to add a header and footer with custom styling to the video frame.
    Header and footer text is centered.
    """
    # Define font, size, and thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    header_footer_color = (146, 28, 173)  # Use ##921CAD for header and footer text color
    footer_bg_color = (146, 28, 173)  # Footer background color ##921CAD (RGB)
    footer_text_color = (255, 255, 255)  # White color for the footer text

    # Get the frame dimensions
    height, width, _ = frame.shape

    # Create space for the header and footer by adding padding around the frame
    header_height = 40
    footer_height = 40
    total_height = height + header_height + footer_height

    # Create a new frame with the additional space
    new_frame = np.zeros((total_height, width, 3), dtype=np.uint8)

    # Copy the original frame (body) into the new frame
    new_frame[header_height:header_height + height, 0:width] = frame

    # Calculate text width and center positions for header and footer
    text_size_header = cv2.getTextSize(header_text, font, font_scale, thickness)[0]
    text_x_header = (width - text_size_header[0]) // 2

    text_size_footer = cv2.getTextSize(footer_text, font, font_scale, thickness)[0]
    text_x_footer = (width - text_size_footer[0]) // 2

    # Add header text (top of the frame, centered)
    cv2.putText(new_frame, header_text, (text_x_header, 30), font, font_scale, header_footer_color, thickness, lineType=cv2.LINE_AA)

    # Add footer background (create a filled rectangle for the footer background)
    footer_start_y = total_height - footer_height
    cv2.rectangle(new_frame, (0, footer_start_y), (width, total_height), footer_bg_color, -1)

    # Add footer text over the background (centered)
    cv2.putText(new_frame, footer_text, (text_x_footer, total_height - 10), font, font_scale, footer_text_color, thickness, lineType=cv2.LINE_AA)

    return new_frame

def process_webcam(output_file="output.mp4"):
    """
    Process webcam or video file, apply object detection, and save annotated output.
    Ensures the video is 640 pixels wide and maintains aspect ratio.
    """
    cap = cv2.VideoCapture("demo.mp4")  # Replace with 0 for the default webcam

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get the original width, height, and fps of the input video
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set the desired width and scale height proportionally to maintain aspect ratio
    target_width = 1080
    aspect_ratio = original_height / original_width
    target_height = int(target_width * aspect_ratio)

    # Define the codec and create VideoWriter object to save output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi format
    out = cv2.VideoWriter(output_file, fourcc, fps, (target_width, target_height + 80))  # Adjust for header and footer

    # Initialize annotators for bounding boxes and labels
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        ret, frame = cap.read()  # Read frame-by-frame
        if not ret:
            break  # Break if no frame is captured

        # Resize the frame to fit the 640px width, maintaining aspect ratio
        frame = cv2.resize(frame, (target_width, target_height))

        # Apply YOLO model on the frame
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Annotate frame with bounding boxes and labels
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Add header and footer to the annotated frame
        annotated_frame = add_header_footer(annotated_frame)

        # Write the annotated frame to the output file
        out.write(annotated_frame)

        # Display the resulting frame
        cv2.imshow("Webcam", annotated_frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break  # Exit on 'q' key press

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Typer command to run webcam processing
@app.command()
def webcam(output_file: str = "output.mp4"):
    """
    Command-line function to start webcam processing.
    """
    typer.echo("Starting webcam processing...")
    process_webcam(output_file)

# Footer: Exit message to inform the user that the process is finished.
if __name__ == "__main__":
    app()
    print("Video processing completed. Output saved to", "output.mp4")
