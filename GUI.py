import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Processing App")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.process_button = QPushButton("Process Image", self)
        self.process_button.clicked.connect(self.processImage)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.process_button)
        self.central_widget.setLayout(self.layout)

    def loadImage(self, filename):
        image = cv2.imread(filename)
        return image

    def processImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                   "Image Files (*.png *.jpg *.bmp *.jpeg);;All Files (*)",
                                                   options=options)

        if file_name:
            image = self.loadImage(file_name)

            # Example: Convert image to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert OpenCV image to QImage with RGB format
            h, w, c = rgb_image.shape
            bytes_per_line = 3 * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Create QPixmap from QImage
            pixmap = QPixmap.fromImage(q_image)

            # Display the processed image
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)


def main():
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
