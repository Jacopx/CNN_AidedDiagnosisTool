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
import PhotoViewer
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
        self.file_name = ''

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
        self.reload_button = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.reload_button.setObjectName("reload_button")
        self.horizontalLayout.addWidget(self.reload_button)

        self.clear_button.setDisabled(True)
        self.mask_button.setDisabled(True)
        self.reload_button.setDisabled(True)

        self.drop_label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.drop_label.setObjectName("drop_label")
        self.horizontalLayout.addWidget(self.drop_label)
        self.dropout_combo = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        self.dropout_combo.setObjectName("dropout_combo")
        self.horizontalLayout.addWidget(self.dropout_combo)
        self.crop_label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.crop_label.setObjectName("crop_label")
        self.horizontalLayout.addWidget(self.crop_label)
        self.crop_combo = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        self.crop_combo.setObjectName("crop_combo")
        self.horizontalLayout.addWidget(self.crop_combo)
        self.iter_label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.iter_label.setObjectName("iter_label")
        self.horizontalLayout.addWidget(self.iter_label)
        self.iter_combo = QtWidgets.QComboBox(self.horizontalLayoutWidget)
        self.iter_combo.setObjectName("iter_combo")
        self.horizontalLayout.addWidget(self.iter_combo)
        self.main_label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.main_label.setObjectName("main_label")
        self.horizontalLayout.addWidget(self.main_label)

        self.legend_viewer = PhotoViewer.PhotoViewer(self.centralwidget)
        self.legend_viewer.setObjectName("legend_viewer")
        self.horizontalLayout.addWidget(self.legend_viewer)

        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)

        self.viewer = PhotoViewer.PhotoViewer(self.centralwidget)
        self.viewer.setGeometry(QtCore.QRect(0, 50, 1941, 981))
        self.viewer.setObjectName("viewer")

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
        # self.actionSavePNG = QtWidgets.QAction(MainWindow)
        # self.actionSavePNG.setObjectName("actionSavePNG")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)
        self.menubar.addAction(self.menuFile.menuAction())
        # self.menuFile.addAction(self.actionSavePNG)
        # self.menuFile.addSeparator()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.clear_button.setText(_translate("MainWindow", "Clear"))
        self.mask_button.setText(_translate("MainWindow", "Mask"))
        self.reload_button.setText(_translate("MainWindow", "Reload"))
        self.drop_label.setText(_translate("MainWindow", "Dropout:"))
        self.crop_label.setText(_translate("MainWindow", "Crop:"))
        self.iter_label.setText(_translate("MainWindow", "Iteration:"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionOpen.setText(_translate("MainWindow", "Open..."))
        self.actionOpen.setStatusTip(_translate("MainWindow", "Open a file"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))
        # self.actionSavePNG.setText(_translate("MainWindow", "Save as PNG"))
        # self.actionSavePNG.setStatusTip(_translate("MainWindow", "Save map with highlights in PNG format"))
        # self.actionSavePNG.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))

    # Function to define correlations to actions
    def define_actions(self):
        # Buttons
        self.clear_button.clicked.connect(self.clear_img)
        self.mask_button.clicked.connect(self.mask_change)
        self.reload_button.clicked.connect(self.generate)

        # Menu Bar
        self.actionOpen.triggered.connect(self.open_file)
        # self.actionSavePNG.triggered.connect(self.save_file)

    # Fill the combobox with proper values
    def fill_combo(self):
        self.crop_combo.addItems(['2240', '4480'])
        self.dropout_combo.addItems(['0.1', '0.01', '0.5'])
        self.iter_combo.addItems(['1', '10', '100', '1000', '10000'])

    # Function triggered when Open command is executed, save the path in global var
    def open_file(self):
        self.file_name = QFileDialog.getOpenFileNames()
        self.generate()

    # Invoked after OPEN or RELOAD are called, actually open the the file and enable the buttons
    def generate(self):
        # Single image selected a new process is started
        if len(self.file_name[0]) > 0:
            crop = int(self.crop_combo.currentText())
            drop = float(self.dropout_combo.currentText())
            iteration = int(self.iter_combo.currentText())

            self.out_file = main.make_prediction(self.file_name[0], crop, drop, iteration)
            self.show_img(self.out_file[0][1])
            self.mask = True
            self.clear_button.setDisabled(False)
            self.mask_button.setDisabled(False)
            self.reload_button.setDisabled(False)
            self.main_label.setText(self.out_file[0][1])
            self.legend_viewer.setPhoto(QtGui.QPixmap(self.out_file[0][1]))

    # Change from mask and no-mask rapidly
    def mask_change(self):
        if self.mask:
            self.show_img(self.out_file[0][0])
            self.main_label.setText(self.out_file[0][0])
            self.mask = False
        else:
            self.show_img(self.out_file[0][1])
            self.main_label.setText(self.out_file[0][1])
            self.mask = True

    # Show image
    def show_img(self, path):
        self.viewer.setPhoto(QtGui.QPixmap(path))

    # Activate drag mode
    def pixInfo(self):
        self.viewer.toggleDragMode()

    # Active image pan
    def photoClicked(self, pos):
        if self.viewer.dragMode() == QtWidgets.QGraphicsView.NoDrag:
            self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))

    # Clear image, deactive buttons and restore label
    def clear_img(self):
        self.clear_button.setDisabled(True)
        self.mask_button.setDisabled(True)
        self.reload_button.setDisabled(True)
        self.main_label.setText('')
        self.viewer.setPhoto()
        self.legend_viewer.setPhoto()

# MAIN
if __name__ == "__main__":
    src.preparation.utils.test_folder()
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())

