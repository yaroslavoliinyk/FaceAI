from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PIL import Image
from modules.AppImage import AppImage, OriginalAppImage
import time
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.APP_WIDTH = 1200
        self.APP_HEIGHT = 750

        self.setWindowTitle("FaceAI")
        self.setFixedWidth(self.APP_WIDTH)
        self.setFixedHeight(self.APP_HEIGHT)
        self.file_dialog = ""
        self.file_path = ""
        self.label_chosen_image = QLabel(self)
        self.image_is_descaled = False
        self.image_is_ready = False
        self.result_img = None
        self.app_image = OriginalAppImage()

        # Controls and labels
        self.main_label = self.__get_label("Face AI", 60, True)
        self.choose_image_label = self.__get_label("Choose image to upscale", 12)
        self.choose_btn = self.__get_choose_btn()
        self.choose_algorithm_label = self.__get_label("Choose upscale algorithm", 12)
        self.algorithm_combobox = self.__get_combobox(AppImage.ALGORITHMS)
        self.run_upscale_label = self.__get_label("Run upscaling", 16, True)
        self.upscale_btn = self.__get_upscale_btn()
        self.original_energy_label = self.__get_label(
            "energy: ?.??", 12, underline=True
        )
        self.original_sharpness_brenner_label = self.__get_label(
            "sharpness: ?.??", 11, underline=True
        )
        self.original_SMD_label = self.__get_label("SMD: ?.??", 11, underline=True)
        self.original_color_hist_btn = self.__get_color_hist_btn()
        self.upscaled_energy_label = self.__get_label(
            "energy: ?.??", 12, underline=True
        )
        self.upscaled_sharpness_brenner_label = self.__get_label(
            "sharpness: ?.??", 11, underline=True
        )
        self.upscaled_SMD_label = self.__get_label("SMD: ?.??", 11, underline=True)
        self.upscaled_color_hist_btn = self.__get_color_hist_btn()
        self.input_image_label = self.__get_label("Input image", 24)
        self.upscaled_image_label = self.__get_label("Upscaled image", 24)
        self.original_image = self.__get_original_image()
        self.upscaled_image = self.__get_upscaled_image()
        self.madeby_label = self.__get_label("made by Yaroslav Oliinyk", 10)

        # Connect buttons to the proper methods
        self.choose_btn.clicked.connect(self.__on_choose_bnt_click)
        self.upscale_btn.clicked.connect(self.__on_upscale_btn_click)
        self.original_color_hist_btn.clicked.connect(self.__on_origin_hist_btn_clicked)
        self.upscaled_color_hist_btn.clicked.connect(
            self.__on_upscaled_hist_btn_clicked
        )

        # Part of Vertical layout
        vbox = QVBoxLayout()
        choose_img_vbox = QVBoxLayout()
        choose_img_vbox.addWidget(self.choose_image_label)
        choose_img_vbox.addWidget(self.choose_btn)

        choose_alg_vbox = QVBoxLayout()
        choose_alg_vbox.addWidget(self.choose_algorithm_label)
        choose_alg_vbox.addWidget(self.algorithm_combobox)

        run_alg_vbox = QVBoxLayout()
        run_alg_vbox.addWidget(self.run_upscale_label)
        run_alg_vbox.addWidget(self.upscale_btn)

        vbox.addLayout(choose_img_vbox)
        vbox.addLayout(choose_alg_vbox)
        vbox.addLayout(run_alg_vbox)

        # Making everything in Grid layout
        layout = QGridLayout()
        layout.addWidget(self.main_label, 0, 4, 1, 2)
        layout.addLayout(vbox, 4, 0, 1, 1)
        layout.addWidget(self.input_image_label, 1, 2, 1, 3)
        layout.addWidget(self.upscaled_image_label, 1, 5, 1, 3)
        layout.addWidget(self.original_image, 2, 2, 3, 3)
        layout.addWidget(self.original_energy_label, 5, 2, 1, 1)
        layout.addWidget(self.original_sharpness_brenner_label, 5, 3, 1, 1)
        layout.addWidget(self.original_SMD_label, 5, 4, 1, 1)
        layout.addWidget(self.original_color_hist_btn, 6, 2, 1, 1)
        layout.addWidget(self.upscaled_image, 2, 5, 3, 3)
        layout.addWidget(self.upscaled_energy_label, 5, 5, 1, 1)
        layout.addWidget(self.upscaled_sharpness_brenner_label, 5, 6, 1, 1)
        layout.addWidget(self.upscaled_SMD_label, 5, 7, 1, 1)
        layout.addWidget(self.upscaled_color_hist_btn, 6, 5, 1, 1)
        layout.addWidget(self.madeby_label, 6, 6, 1, 2)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def __get_label(
        self, text, font_size, is_bold=False, underline=False, align_center=True
    ):
        label = QLabel(text, self)
        font = QtGui.QFont("verdana", font_size, QtGui.QFont.Light)
        if is_bold:
            font.setBold(True)
        if align_center:
            label.setAlignment(Qt.AlignCenter)
        font.setUnderline(underline)
        label.setFont(font)
        return label

    def __get_combobox(self, algorithms):
        combobox = QComboBox()
        combobox.addItems(algorithms)
        return combobox

    def __get_color_hist_btn(self):
        button = QPushButton("Colors Histogram", self)
        button.setToolTip("Colors Histogram")
        return button

    def __get_original_image(self):
        label_name = "Image"
        label = QLabel(label_name, self)
        image_pixmap = QPixmap(self.app_image.image_path)
        image_pixmap = image_pixmap.scaled(
            600, self.app_image.app_image_height * 600 / self.app_image.app_image_width
        )
        label.setPixmap(image_pixmap)
        label.setAlignment(Qt.AlignCenter)
        return label

    def __get_upscaled_image(self):
        label_name = "Image"
        label = QLabel(label_name, self)
        image_pixmap = QPixmap(self.app_image.upscaled_app_image.image_path)
        image_pixmap = image_pixmap.scaled(
            600,
            self.app_image.upscaled_app_image.app_image_height
            * 600
            / self.app_image.upscaled_app_image.app_image_width,
        )
        label.setPixmap(image_pixmap)
        label.setAlignment(Qt.AlignCenter)
        return label

    def __get_choose_btn(self):
        button = QPushButton("Choose..", self)
        button.setToolTip("Choose..")
        return button

    def __get_upscale_btn(self):
        button = QPushButton("Upscale", self)
        button.setToolTip("Upscale")
        return button

    def __on_choose_bnt_click(self):
        self.upscale_btn.setHidden(False)
        path = QFileDialog.getOpenFileName(self, "Select Directory")
        image_path = path[0]
        image_path.encode("unicode_escape")
        self.app_image.image_path = image_path
        self.__update_app_images()

    def __on_upscale_btn_click(self):
        app_image = self.app_image
        app_image.upscale(self.algorithm_combobox.currentText())
        self.__update_app_images()
        QMessageBox.about(
            self, "Successfull upscaling", "The image upscaling has been completed.\n"
        )

    # Update what's going on on the screen now(if it's descaled or original)
    def __update_app_images(self):
        original_image_pixmap = QPixmap(self.app_image.image_path)
        original_image_pixmap = original_image_pixmap.scaled(
            600, self.app_image.app_image_height * 600 / self.app_image.app_image_width
        )
        self.original_image.setPixmap(original_image_pixmap)
        upscaled_image_pixmap = QPixmap(self.app_image.upscaled_app_image.image_path)
        upscaled_image_pixmap = upscaled_image_pixmap.scaled(
            600,
            self.app_image.upscaled_app_image.app_image_height
            * 600
            / self.app_image.upscaled_app_image.app_image_width,
        )
        self.upscaled_image.setPixmap(upscaled_image_pixmap)
        self.__update_measurements()

    def __update_measurements(self):
        self.original_energy_label.setText(
            "energy: " + str(self.app_image.get_energy())
        )
        self.original_sharpness_brenner_label.setText(
            "sharpness: " + str(self.app_image.get_sharpness_brenner())
        )
        self.original_SMD_label.setText("SMD: " + str(self.app_image.get_SMD()))
        self.upscaled_energy_label.setText(
            "energy: " + str(self.app_image.upscaled_app_image.get_energy())
        )
        self.upscaled_sharpness_brenner_label.setText(
            "sharpness: "
            + str(self.app_image.upscaled_app_image.get_sharpness_brenner())
        )
        self.upscaled_SMD_label.setText(
            "SMD: " + str(self.app_image.upscaled_app_image.get_SMD())
        )

    def __on_origin_hist_btn_clicked(self):
        hist_image = self.app_image.get_histogram()
        img = Image.fromarray(hist_image)
        img.save("original_histogram.jpg")
        self.original_window = HistogramWindow()
        self.original_window.set_picture("original_histogram.jpg")
        print("original window", self.original_window)
        self.original_window.show()

    def __on_upscaled_hist_btn_clicked(self):
        hist_image = self.app_image.upscaled_app_image.get_histogram()
        img = Image.fromarray(hist_image)
        img.save("upscaled_histogram.jpg")
        self.upscaled_window = HistogramWindow()
        self.upscaled_window.set_picture("upscaled_histogram.jpg")
        print("upscaled window", self.upscaled_window)
        self.upscaled_window.show()


class HistogramWindow(QWidget):
    def init(self):
        super().__init__()
        self.title = "Histogram Viewer"
        self.setWindowTitle(self.title)

    def set_picture(self, path):
        pixmap = QPixmap(path)
        layout = QVBoxLayout()
        label = QLabel("Histogram")
        label.setPixmap(pixmap)
        layout.addWidget(label)
        self.setLayout(layout)


app = QApplication(sys.argv)
window = MainWindow()
app.setStyle("Fusion")

dark_palette = QPalette()
dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
dark_palette.setColor(QPalette.WindowText, Qt.white)
dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
dark_palette.setColor(QPalette.ToolTipText, Qt.white)
dark_palette.setColor(QPalette.Text, Qt.white)
dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
dark_palette.setColor(QPalette.ButtonText, Qt.white)
dark_palette.setColor(QPalette.BrightText, Qt.red)
dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
dark_palette.setColor(QPalette.HighlightedText, Qt.black)
app.setPalette(dark_palette)

window.show()
app.exec_()
