import select
import socket
import time
from typing import List

import numpy as np
from epics import caget_many


class ROIError(Exception):
    pass


class AWACamera:
    awa_camera_host = "127.0.0.1"
    awa_pg_port = 2019
    awa_nifg_port = 2029

    channel_names = ["FG_FWHMX", "FG_FWHMY", "FG_FWHML", "FG_CX", "FG_CY", "FG_IMG"]

    def __init__(self, camera_type):
        self.initialized = False
        self.testing = True if camera_type is None else False

        # initialize connections
        if not self.testing:
            # lazy import if not testing
            from win32com import client
            
            # no idea what this is -- used for gating measurement
            self.usb_dio_client = client.Dispatch("USBDIOCtrl.Application")

            # cameras
            self.camera_type = camera_type
            if camera_type == "NIFG":
                self.camera_app = client.Dispatch("NIFGCtrl")
                self.camera_port = self.awa_nifg_port
            elif camera_type == "AWAPG":
                self.camera_app = client.Dispatch("AWAPGCamera.application")
                self.camera_port = self.awa_pg_port
            else:
                raise ValueError("camera type not available")

            self.camera_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.camera_client.connect(("127.0.0.1", self.camera_port))
            self.camera_client.setblocking(False)

    def get_roi(self):
        x1 = self.camera_app.ROIX1
        x2 = self.camera_app.ROIX2
        y1 = self.camera_app.ROIY1
        y2 = self.camera_app.ROIY2

        if np.any(np.array((np.abs(x2 - x1), np.abs(y2 - y1))) < 20):
            raise ROIError("ROI is not large enough!")

        return np.array(((x1, y1), (x2, y2)))

    def get_raw_image(self):
        if self.camera_type == "NIFG":
            return np.array(self.camera_app.GetImage())
        elif self.camera_type == "AWAPG":
            return np.array(self.camera_app.GetImage)

    def get_image_data(self, keys: List[str]):
        a = self.camera_client.recv(1024)
        b = "".join(chr(x) for x in a)
        c = eval(b)
        return np.array([c[key] for key in keys])

    def get_measurement(self, epics_pvs: list = None):
        """
        get new image and charge data

        Arguments
        ---------

        note - calculations of centroids and FWHM etc. are based on a region of
        interest, which might be changed by the user!

        Connect to camera broadcasting TCP port for notifications
        If it is timed out, then just download and return whatever
        image available
        In order to avoid the complication of TCP buffer cleanup
        we will simply close the connection and reopen it.
        """

        roi = None
        image_data = []
        img = None
        image_data_keys = self.observable_names[:-1]

        epics_pvs = epics_pvs or {}
        epics_measurements = {}

        if not self.testing:
            ready = select.select([self.camera_client], [], [], 2)

            if ready[0]:
                # gate measurement
                self.usb_dio_client.SetReadyState(2, 1)

                while True:
                    try:
                        # get image data and stats
                        image_data = self.get_image_data(image_data_keys)
                        img = self.get_raw_image()

                        # get additional epics measurements
                        epics_measurements = caget_many(epics_pvs)

                        # get ROI
                        roi = self.get_roi()

                        # if successful
                        break

                    except SyntaxError:
                        RuntimeWarning("sleeping!")
                        time.sleep(0.1)

                # set state to false
                self.usb_dio_client.SetReadyState(2, 0)

        else:
            image_data = np.ones(5)
            img = np.ones(10, 10)

            # get ROI
            roi = np.array(((0, 0), (10, 10)))

        output = {self.observable_names[-1]: img, "ROI": roi}
        for i, name in enumerate(image_data_keys):
            output[name] = image_data[i]

        # append epics measurements to data
        output = output.update(epics_measurements)

        return output
