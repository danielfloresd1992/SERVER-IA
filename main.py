import sys

# SERVER
from src.app.server import Server_services

#
# CORE 
from src.gui.q_application import AppSingletonGui

# GUI
from src.gui.window_main import Main_Window




def main():
    
    server = Server_services(port=9000)
    
    app = AppSingletonGui.initialize()
    main_window = Main_Window()
    
    main_window.btn_server_init.clicked.connect(server.start)
    main_window.btn_server_stop.clicked.connect(server.stop)
    
    main_window.show()
    app.exec()



if __name__ == '__main__':
    sys.exit(main())