# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          GUI_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/GUI_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#                                usage: GUI.py                              *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
import sys
import src.preparation.utils
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import main

COOL_DOWN_TIME = 1


# MainWindows class
class Ui_MainWindow(object):
    # Function executed and the instantiation of class
    def __init__(self):
        self.setupUi(MainWindow)
        self.define_actions()
        self.fill_combo()
        self.mask = False
        self.out_file = []

    # AUTO GENERATED GRAPHICAL
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 1921, 51))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.clear_button = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.clear_button.setObjectName("clear_button")
        self.horizontalLayout.addWidget(self.clear_button)
        self.mask_button = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.mask_button.setObjectName("mask_button")
        self.horizontalLayout.addWidget(self.mask_button)
        self.dropout_combo = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        self.dropout_combo.setObjectName("dropout_combo")
        self.horizontalLayout.addWidget(self.dropout_combo)
        self.crop_combo = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        self.crop_combo.setObjectName("crop_combo")
        self.horizontalLayout.addWidget(self.crop_combo)
        self.main_label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.main_label.setObjectName("main_label")
        self.horizontalLayout.addWidget(self.main_label)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(0, 89, 1920, 941))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image.sizePolicy().hasHeightForWidth())
        self.image.setSizePolicy(sizePolicy)
        self.image.setText("")
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.image.setObjectName("image")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSavePNG = QtWidgets.QAction(MainWindow)
        self.actionSavePNG.setObjectName("actionSavePNG")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSavePNG)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.clear_button.setText(_translate("MainWindow", "Clear"))
        self.mask_button.setText(_translate("MainWindow", "Mask"))
        self.image.setText(_translate("MainWindow", "Image"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionOpen.setText(_translate("MainWindow", "Open..."))
        self.actionOpen.setStatusTip(_translate("MainWindow", "Open a file"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionSavePNG.setText(_translate("MainWindow", "Save as PNG"))
        self.actionSavePNG.setStatusTip(_translate("MainWindow", "Save map with highlights in PNG format"))
        self.actionSavePNG.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))

    # Function to define correlations to actions
    def define_actions(self):
        # Buttons
        self.clear_button.clicked.connect(self.clear_img)
        self.mask_button.clicked.connect(self.mask_change)

        # Menu Bar
        self.actionOpen.triggered.connect(self.open_file)
        self.actionSavePNG.triggered.connect(self.save_file)

    def fill_combo(self):
        self.crop_combo.addItems(['2240', '4480'])
        self.dropout_combo.addItems(['0.1', '0.01', '0.5'])

    # Open selected file after actionOpen trigger
    # Single file selection drive to a single evaluation.
    # Using multiple selection will start the evaluation of the first immidiately and than the others
    def open_file(self):
        file_name = QFileDialog.getOpenFileNames()

        # Single image selected a new process is started
        if len(file_name[0]) > 0:
            crop = int(self.crop_combo.currentText())
            drop = float(self.dropout_combo.currentText())
            self.out_file = main.make_prediction(file_name[0], crop, drop, 1)
            self.show_img(self.out_file[0][1])

    def mask_change(self):
        if self.mask:
            self.show_img(self.out_file[0][0])
            self.mask = False
        else:
            self.show_img(self.out_file[0][1])
            self.mask = True

    # Show image
    def show_img(self, path):
        width = self.centralwidget.width()
        height = self.centralwidget.height()

        pix = QtGui.QPixmap(path)

        if pix.height() >= height:
            pix = pix.scaledToHeight(height)
        elif pix.width() >= width:
            pix = pix.scaledToWidth(width)
        else:
            pix = pix.scaledToHeight(720)
            pix = pix.scaledToWidth(1280)

        self.image.setPixmap(pix)

    # Clear image
    def clear_img(self):
        self.image.setPixmap(QtGui.QPixmap(""))

    # Save file to PNG
    def save_file(self):
        msg = QMessageBox()
        msg.setWindowTitle("Save file")
        msg.setText("Save file to PNG")
        msg.setInformativeText("The file will be saved in: ")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        msg.setDefaultButton(QMessageBox.Ok)
        msg.buttonClicked.connect(self.save_popup)
        x = msg.exec_()

    # Manage selection of the user
    def save_popup(self, i):
        if i.text() == 'OK':
            print('File saved...')
        else:
            print('File NOT saved')


if __name__ == "__main__":
    src.preparation.utils.test_folder()
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())

