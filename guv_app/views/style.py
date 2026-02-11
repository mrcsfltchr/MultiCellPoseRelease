from PyQt6 import QtGui

def stylesheet():
    return """
        QToolTip { 
            background-color: black; 
            color: white; 
            border: black solid 1px
        }
        QComboBox {
            color: white;
            background-color: rgb(40,40,40);
        }
        QComboBox::item:enabled { 
            color: white;
            background-color: rgb(40,40,40);
            selection-color: white;
            selection-background-color: rgb(50,100,50);
        }
        QComboBox::item:!enabled {
            background-color: rgb(40,40,40);
            color: rgb(100,100,100);
        }
        QScrollArea > QWidget > QWidget {
            background: transparent;
            border: none;
            margin: 0px 0px 0px 0px;
        } 
        QGroupBox { 
            border: 1px solid white; 
            color: rgb(255,255,255);
            border-radius: 6px;
            margin-top: 8px;
            padding: 0px 0px;
        }            
        QPushButton:pressed {
            text-align: center; 
            background-color: rgb(150,50,150); 
            border-color: white;
            color:white;
        }
        QPushButton:!pressed {
            text-align: center; 
            background-color: rgb(50,50,50);
            border-color: white;
            color:white;
        }
        QPushButton:disabled {
            text-align: center; 
            background-color: rgb(30,30,30);
            border-color: white;
            color:rgb(80,80,80);
        }
    """

class DarkPalette(QtGui.QPalette):
    def __init__(self):
        super().__init__()
        self.setup()

    def setup(self):
        self.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(40, 40, 40))
        self.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(34, 27, 24))
        self.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(53, 50, 47))
        self.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(53, 50, 47))
        self.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor(255, 0, 0))
        self.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(42, 130, 218))
        self.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(42, 130, 218))
        self.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(0, 0, 0))
        self.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text,
                      QtGui.QColor(128, 128, 128))
        self.setColor(
            QtGui.QPalette.ColorGroup.Disabled,
            QtGui.QPalette.ColorRole.ButtonText,
            QtGui.QColor(128, 128, 128),
        )
        self.setColor(
            QtGui.QPalette.ColorGroup.Disabled,
            QtGui.QPalette.ColorRole.WindowText,
            QtGui.QColor(128, 128, 128),
        )
