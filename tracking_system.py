from PyQt5 import uic
from PyQt5.QtWidgets import * 
from PyQt5.QtCore import QThreadPool, QRectF, QTimer
from PyQt5.QtGui import QColor
import pyqtgraph as pg 
import numpy as np
from PIL import Image
import sys
############ import script in the below form #############
from drone_anchor import *

############import map image####################
background_raw = Image.open('itbt_lab.png')
background_image = np.rot90(np.array(background_raw), k = 3)

state_text = '''
<p style='text-align:left;color:black'> <b>&lt; Object Coordinate &gt;</b></p>
<p style='color:black'>
UAV 1 : [ %.2f , %.2f , %.2f ] m<br>
UAV 2 : [ %.2f , %.2f , %.2f ] m<br>
UAV 3 : [ %.2f , %.2f , %.2f ] m<br>
UAV 4 : [ %.2f , %.2f , %.2f ] m<br>
UAV 5 : [ %.2f , %.2f , %.2f ] m<br>
UGV &nbsp;&nbsp;&nbsp;: [ %.2f , %.2f , 0.00 ] m</p>
<p style='color:black'>Positioning UGV using <font style="color:red">Multilateration</font></p>
'''

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        #Load the UI Page
        uic.loadUi('pyqtgraph_gui_noplot.ui', self)

        ############ map image settings ####################
        # set the width and height of image in real distance(meter) [width, height]
        self.map_size = [8.1,8.7]
        # set the real coord of left bottom angular point of image [x, y]
        self.map_coord = [0,0]
        ####################################################
 
        ############ run imported code with thread #############
        self.threadpool = QThreadPool()
        self.threadpool.start(self.run_code)
        ####################################################

        # coordinate variables used in graph
        self.uav_paint = 0
        self.ugv_paint = 0
        self.vehicle = 0

        # <graphWidget setting>
        # set range with padding
        self.graphWidget.setXRange(0, self.map_size[0], padding = 0.03)
        self.graphWidget.setYRange(0, self.map_size[1], padding = 0.03)
        # add grid
        self.graphWidget.showGrid(x = True, y = True, alpha = 0.3)
        # add labels
        self.graphWidget.setLabels(left = 'y [m]', bottom = 'x [m]')
        # add graph background color to transparant
        self.graphWidget.setBackground(background = None)
        # add map image as background
        self.map_image = pg.ImageItem(background_image)
        self.map_size = QRectF(self.map_coord[0],self.map_coord[1],self.map_size[0],self.map_size[1])
        self.map_image.setRect(self.map_size)
        self.graphWidget.addItem(self.map_image)
        # set the legend 
        # ScatterPlotItem does not support legend so by doing this way we can make legends
        self.graphWidget.addLegend(offset = (-1,1), labelTextColor = 'k')
        self.graphWidget.plot([-10, -10], pen ='b', symbolBrush ='b',
                          symbolPen ='b', symbol ='o', symbolSize = 10, name ='UAV')
        self.graphWidget.plot([-10, -10], pen ='b', symbolBrush ='r',
                          symbolPen ='r', symbol ='s', symbolSize = 15, name ='UGV')
        # </graphWidget setting>


        # set window title
        self.setWindowTitle("Wireless System Laboratory")
        # set state text and title text color to black
        textcolor = QColor(0,0,0,255)
        self.state_box.setTextColor(textcolor)
        # set title of GUI
        self.title.setHtml("<p style='color:black; text-align:center; font-size:30pt'>UGV Tracking System using UAVs</p>")

        # set update frequency of graph, state, plot and status bar 
        self.mytimer1 = QTimer()
        self.mytimer1.start(100) # 1000 is 1 sec
        self.mytimer1.timeout.connect(self.graph_update)
        self.mytimer1.timeout.connect(self.state_update)
        self.mytimer1.timeout.connect(self.time_update)

    def graph_update(self):
        # clear graph
        self.graphWidget.removeItem(self.uav_paint)
        self.graphWidget.removeItem(self.ugv_paint)

        ####################################################
        # get variable from imported code
        code_data.trans_var()
        self.vehicle = np.array(code_data.xyz)
        # set paint setting (uav to blue circle, ugv to red square) (notice. pos = np.array([[~,~]])
        self.uav_paint = pg.ScatterPlotItem(pos = self.vehicle[:-1,:2], symbol = 'o', brush='b', pen='b', size = '10')
        self.ugv_paint = pg.ScatterPlotItem(pos = self.vehicle[-1:,:2], symbol = 's', brush='r', pen='r', size = '15')
        ####################################################

        # paint graph
        self.graphWidget.addItem(self.ugv_paint)
        self.graphWidget.addItem(self.uav_paint)

    def state_update(self):
        # clear the textedit
        self.state_box.clear()
        # insert text in textedit
        if self.vehicle[5][0] == -10.00:
            self.state_box.insertHtml(state_text %(self.vehicle[0][0], self.vehicle[0][1],self.vehicle[0][2],
            self.vehicle[1][0], self.vehicle[1][1],self.vehicle[1][2],
            self.vehicle[2][0], self.vehicle[2][1],self.vehicle[2][2],
            self.vehicle[3][0], self.vehicle[3][1],self.vehicle[3][2],
            self.vehicle[4][0], self.vehicle[4][1],self.vehicle[4][2],
            0.00, 0.00))
        else:
            self.state_box.insertHtml(state_text %(self.vehicle[0][0], self.vehicle[0][1],self.vehicle[0][2],
            self.vehicle[1][0], self.vehicle[1][1],self.vehicle[1][2],
            self.vehicle[2][0], self.vehicle[2][1],self.vehicle[2][2],
            self.vehicle[3][0], self.vehicle[3][1],self.vehicle[3][2],
            self.vehicle[4][0], self.vehicle[4][1],self.vehicle[4][2],
            self.vehicle[5][0], self.vehicle[5][1]))

    def time_update(self):
        self.statusbar.showMessage(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    def run_code(self):
        swarm.parallel(run_sequence, args_dict=seq_args)

if __name__ == '__main__':      
    ############# main of imported code ################
    cflib.crtp.init_drivers(enable_debug_driver=False)
    factory = CachedCfFactory(rw_cache='./cache')
    swarm = Swarm_change.Swarm(uris, factory=factory)
    swarm.open_links()
    swarm.parallel(log_download)
    print('Waiting for parameters to be downloaded...')
    swarm.parallel(wait_for_param_download)
    # call wanted variables from imported code 
    code_data = trans_self()
    ####################################################

    input('Enter to start')

    # run gui
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

