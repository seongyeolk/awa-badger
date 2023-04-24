import socket
import time
from typing import List

import numpy as np
import select
from epics import caget
from win32com import client


class ROIError(Exception):
    pass


class AWACamera:
    awa_camera_host = "127.0.0.1"
    awa_pg_port = 2019
    awa_nifg_port = 2029

    def __init__(self, camera_type, testing=False):
        self.initialized = False
        self.testing = testing

        # initialize connections
        if not self.testing:
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
        if not self.testing:
            x1 = self.camera_app.ROIX1
            x2 = self.camera_app.ROIX2
            y1 = self.camera_app.ROIY1
            y2 = self.camera_app.ROIY2
        else:
            x1, x2, y1, y2 = 0, 100, 0, 100

        if np.any(np.array((np.abs(x2 - x1), np.abs(y2 - y1))) < 20):
            raise ROIError("ROI is not large enough!")

        return np.array(((x1, y1), (x2, y2)))

    def get_raw_image(self):
        if not self.testing:
            if self.camera_type == "NIFG":
                return np.array(self.camera_app.GetImage())
            elif self.camera_type == "AWAPG":
                return np.array(self.camera_app.GetImage)
        else:
            return np.random.rand(20, 20)

    def get_image_data(self, keys: List[str]):
        if not self.testing:
            a = self.camera_client.recv(1024)
            b = "".join(chr(x) for x in a)
            c = eval(b)
            return np.array([c[key] for key in keys])
        else:
            return np.ones(len(keys))

    def get_charge(self):
        if not self.testing:
            return np.abs(np.array([caget(f"AWAICTMon:Ch{i}") for i in range(1, 5)]))
        else:
            return np.ones(4)

    def get_measurement(self, target_charge=-1, charge_deviation=0.1, n_samples=1):
        """
        get new image and charge data

        Arguments
        ---------
        target_charge : float, optional
            Target charge for valid observation in nC, if negative ignore.
            Default: -1 (ignore)

        charge_deviation : float, optional
            Fractional deviation from target charge on ICT1 allowed for valid
            observation. Default: 0.1

        n_samples : int
            Number of samples to take

        note - calculations of centroids and FWHM etc. are based on a region of
        interest, which might be changed by the user!

        Connect to camera broadcasting TCP port for notifications
        If it is timed out, then just download and return whatever
        image available
        In order to avoid the complication of TCP buffer cleanup
        we will simply close the connection and reopen it.
        """

        n_shots = 0
        roi = None
        img = []
        charge = np.empty((n_samples, 4))

        image_data = np.empty((n_samples, 5))
        image_data_keys = ["FWHMX", "FWHMY", "FWHML", "CX", "CY"]

        while n_shots < n_samples:
            if not self.testing:
                ready = select.select([self.camera_client], [], [], 2)

                if ready[0]:
                    # gate measurement
                    self.usb_dio_client.SetReadyState(2, 1)

                    # check charge on ICT1 is within bounds or charge bounds is not
                    # specified (target charge < 0)
                    ict1_charge = self.get_charge()[0]
                    if (
                        np.abs(ict1_charge - target_charge)
                        < np.abs(charge_deviation * target_charge)
                    ) or (target_charge < 0):
                        try:
                            # get image data and stats
                            image_data[n_shots] = self.get_image_data(image_data_keys)
                            img += [self.get_raw_image()]

                            # get charge
                            charge[n_shots] = self.get_charge()

                            # get ROI
                            roi = self.get_roi()
                            n_shots += 1

                        except SyntaxError:
                            RuntimeWarning("sleeping!")
                            time.sleep(0.1)

                    else:
                        # if we are considering charge limits then print a warning
                        if target_charge > 0:
                            RuntimeWarning(
                                f"ICT1 charge:{ict1_charge} nC" f" is outside target range"
                            )
                            time.sleep(0.1)

                    # set state to false
                    self.usb_dio_client.SetReadyState(2, 0)
            else:
                image_data[n_shots] = self.get_image_data(image_data_keys)
                img += [self.get_raw_image()]

                # get charge
                charge[n_shots] = self.get_charge()

                # get ROI
                roi = self.get_roi()
                n_shots += 1

        img = np.array(img)

        output = {"charge": charge, "raw_images": img, "ROI": roi}
        for i, name in enumerate(image_data_keys):
            output[name] = image_data[:, i]

        return output
