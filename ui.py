from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PIL import Image
from modules.AppImage import AppImage

import sys
import time

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.APP_WIDTH                  = 1200
        self.APP_HEIGHT                 = 750
        self.INITIAL_IMAGE_NAME         = "imgs/empty-avatar.png"

        self.setWindowTitle("FaceAI")
        self.setFixedWidth(self.APP_WIDTH)
        self.setFixedHeight(self.APP_HEIGHT)
        self.file_dialog                 = ""
        self.file_path                   = ""
        self.label_chosen_image          = QLabel(self)
        self.image_is_descaled           = False
        self.image_is_ready              = False
        self.result_img                  = None
        self.app_image                   = AppImage(self.INITIAL_IMAGE_NAME)        

        # Controls and labels
        self.main_label                  = self.__get_label("Face AI", 60, True)
        self.choose_image_label          = self.__get_label("Choose image to upscale", 20)
        self.choose_btn                  = self.__get_choose_btn()
        self.choose_algorithm_label      = self.__get_label("Choose upscale algorithm", 20)
        self.algorithm_combobox          = self.__get_combobox(AppImage.ALGORITHMS)
        self.choose_algorithm_label      = self.__get_label("Run upscaling", 20)
        self.upscale_btn                 = self.__get_upscale_btn()
        self.original_entropy_label      = self.__get_label("Entropy: ", 15)
        self.original_mse_label          = self.__get_label("MSE: ", 15)
        self.original_ssi_label          = self.__get_label("SSI: ", 15)
        self.original_color_hist_btn     = self.__get_color_hist_btn()
        self.upscaled_entropy_label      = self.__get_label("Entropy: ", 15)
        self.upscaled_mse_label          = self.__get_label("MSE: ", 15)
        self.upscaled_ssi_label          = self.__get_label("SSI: ", 15)
        self.upscaled_color_hist_btn     = self.__get_color_hist_btn()
        self.input_image_label           = self.__get_label("Input image", 24)
        self.upscaled_image_label        = self.__get_label("Upscaled image", 24)
        self.original_image              = self.__get_original_image(self.app_image)
        self.upscaled_image              = self.__get_upscaled_image(self.app_image)
        self.estimate_original_btn       = self.__get_estimate_btn()
        self.estimate_original_label     = self.__get_estimate_label()
        self.estimate_upscaled_btn       = self.__get_estimate_btn()
        self.estimate_upscaled_label     = self.__get_estimate_label()
        self.descale_checkbox            = self.__get_descale_checkbox()
        self.madeby_label                = self.__get_label("made by Yaroslav Oliinyk", 10)

        # Connect buttons to the proper methods
        self.choose_btn.clicked.connect(self.__on_choose_bnt_click)
        #self.original_color_hist_btn.clicked.connect(self.__on_original_color_hist_bnt_clicked)
        #self.upscaled_color_hist_btn.clicked.connect(self.__on_upscaled_color_hist_bnt_clicked)
        self.estimate_original_btn.clicked.connect(self.__on_estimate_original_btn_click)
        self.estimate_upscaled_btn.clicked.connect(self.__on_estimate_upscaled_btn_click)
        self.upscale_btn.clicked.connect(self.__on_upscale_btn_click)
        self.descale_checkbox.toggled.connect(self.__on_toggled_descale_chbx)

        # Making everything in Grid layout
        layout = QGridLayout()
        layout.addWidget(self.main_label, 0, 4, 1, 2)                  
        layout.addWidget(self.choose_image_label, 1, 0, 1, 2)          
        layout.addWidget(self.choose_btn, 2, 0, 1, 1)                  
        layout.addWidget(self.input_image_label, 1, 2, 1, 3)           
        layout.addWidget(self.upscaled_image_label, 1, 5, 1, 3)        
        layout.addWidget(self.original_image, 2, 2, 3, 3)        
        layout.addWidget(self.upscaled_image, 2, 5, 3, 3)              
        layout.addWidget(self.upscale_btn, 5, 2, 1, 1)                 
        layout.addWidget(self.estimate_original_btn, 5, 3, 1, 1)       
        layout.addWidget(self.estimate_original_label, 5, 4, 1, 1)     
        layout.addWidget(self.estimate_upscaled_btn, 5, 6, 1, 1)       
        layout.addWidget(self.estimate_upscaled_label, 5, 7, 1, 1)     
        layout.addWidget(self.descale_checkbox, 6, 1, 1, 1)            
        layout.addWidget(self.madeby_label, 6, 6, 1, 2)                

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


    def __get_label(self, text, font_size, is_bold=False, align_center=True):
        label    = QLabel(text, self)
        font = QtGui.QFont('verdana', font_size, QtGui.QFont.Light)
        if(is_bold):
            font.setBold(True)
        if(align_center):
            label.setAlignment(Qt.AlignCenter)
        label.setFont(font)
        return label


    def __get_combobox(self, algorithms):
        pass


    def __get_color_hist_btn(self):
        pass


    def __get_original_image(self, app_image):
        label_name   = "Image"
        label        = QLabel(label_name, self)
        image_pixmap = QPixmap(app_image.get_image_path())
        
        image_pixmap = image_pixmap.scaled(600, app_image.get_app_image_height()*600/app_image.get_app_image_width())
        label.setPixmap(image_pixmap)
        label.setAlignment(Qt.AlignCenter)
        return label


    def __get_upscaled_image(self, app_image):
        label_name   = "Image"
        label        = QLabel(label_name, self)
        if(app_image.get_upscaled_image_path() == None):
            image_pixmap = QPixmap(app_image.get_image_path())
        else:
            image_pixmap = QPixmap(app_image.get_upscaled_image_path())
        image_pixmap = image_pixmap.scaled(600, app_image.get_app_image_height()*600/app_image.get_app_image_width())
        label.setPixmap(image_pixmap)
        label.setAlignment(Qt.AlignCenter)
        return label


    def __get_choose_btn(self):
        button = QPushButton('Choose..', self)
        button.setToolTip('Choose..')
        return button
        

    def __get_estimate_label(self):
        self.estimated_score = "?.??"
        font                 = QFont("Arial", 18)
        font = QFont('Terminus', 25, QtGui.QFont.Light)
        label                = QLabel(self.estimated_score, self)
        font.setBold(True)
        font.setUnderline(True)
        label.setFont(font)
        label.setAlignment(Qt.AlignCenter)
        return label


    def __get_estimate_btn(self):
        button = QPushButton('Estimate', self)
        button.setToolTip('Estimate')
        return button


    def __get_descale_checkbox(self):
        chbx     = QCheckBox("5% descale")
        return chbx


    def __get_upscale_btn(self):
        button = QPushButton('Upscale', self)
        button.setToolTip('Upscale')
        button.setHidden(True)
        return button


    def __on_choose_bnt_click(self):
        self.upscale_btn.setHidden(False)
        path                    = QFileDialog.getOpenFileName(self, "Select Directory")
        image_path              = path[0]
        image_path.encode('unicode_escape')
        self.descale_checkbox.setChecked(False)
        self.app_image          = AppImage(image_path)        
        self.__update_app_images(self.app_image)


    def __on_upscale_btn_click(self):
        app_image = self.app_image
        app_image.upscale()
        self.__update_app_images(app_image)
        QMessageBox.about(self, "Successfull upscaling", "The image upscaling has been completed.\n")


    def __on_estimate_original_btn_click(self):
        app_image = self.app_image
        score     = None
        if(app_image.get_on_screen_descaled()):
            score = app_image.assess_image(app_image.get_descaled_image_path())
        else:
            score = app_image.assess_image(app_image.get_image_path())

        self.estimate_original_label.setText(str(score))


    def __on_estimate_upscaled_btn_click(self):
        app_image = self.app_image
        score     = None
        if(app_image.get_on_screen_descaled()):
            score = app_image.assess_image(app_image.get_upscaled_descaled_image_path())
        else:
            score = app_image.assess_image(app_image.get_upscaled_image_path())

        self.estimate_upscaled_label.setText(str(score))


    def __on_toggled_descale_chbx(self):
        app_image = self.app_image
        if(self.descale_checkbox.isChecked()):
            app_image.downscale_minus_5()
        else:
            app_image.set_on_screen_descaled(False)
        
        self.__update_app_images(app_image)

    # Update what's going on on the screen now(if it's descaled or original)
    def __update_app_images(self, app_image):
        if(app_image.get_on_screen_descaled()):
            original_image_pixmap = QPixmap(app_image.get_descaled_image_path())
            upscaled_image_pixmap = QPixmap(app_image.get_upscaled_descaled_image_path())
            self.estimate_original_label.setText(app_image.assess_image(app_image.get_descaled_image_path()))
            self.estimate_upscaled_label.setText(app_image.assess_image(app_image.get_upscaled_descaled_image_path()))
        else:
            original_image_pixmap = QPixmap(app_image.get_image_path())
            upscaled_image_pixmap = QPixmap(app_image.get_upscaled_image_path())
            self.estimate_original_label.setText(app_image.assess_image(app_image.get_image_path()))
            self.estimate_upscaled_label.setText(app_image.assess_image(app_image.get_upscaled_image_path()))

        original_image_pixmap = original_image_pixmap.scaled(600, app_image.get_app_image_height()*600/app_image.get_app_image_width())
        upscaled_image_pixmap = upscaled_image_pixmap.scaled(600, app_image.get_app_image_height()*600/app_image.get_app_image_width())
        self.original_image.setPixmap(original_image_pixmap)
        self.upscaled_image.setPixmap(upscaled_image_pixmap)


    def __image_assessment(self, img_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNNIQAnet(ker_size=7,
                        n_kers=50,
                        n1_nodes=800,
                        n2_nodes=800).to(device)

        model.load_state_dict(torch.load("image_assessment_CNNIQA/models/CNNIQA-LIVE"))
        im = Image.open(img_path).convert('L')
        patches = NonOverlappingCropPatches(im, 32, 32)
        model.eval()
        with torch.no_grad():
            patch_scores = model(torch.stack(patches).to(device))
            score        = 50. - model(torch.stack(patches).to(device)).mean()
            return score



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