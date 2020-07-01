#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python -*-

"""
 @file ImageToObjectPrediction.py
 @brief predict object by chainer
 @date $Date$


"""
import sys

# Import RTM module
import RTC
import OpenRTM_aist
import network


# Import Service implementation class
# <rtc-template block="service_impl">

# </rtc-template>

# Import Service stub modules
# <rtc-template block="consumer_import">
# </rtc-template>

import numpy as np
import chainer.functions as F
from chainer import serializers, Variable
from googlenet import GoogLeNet
import cv2
from PIL import Image

sys.path.append(".")

# This module's spesification
# <rtc-template block="module_spec">
imagetoobjectprediction_spec = [
    "implementation_id", "ImageToObjectPrediction",
    "type_name", "ImageToObjectPrediction",
    "description", "predict image by chainer",
    "version", "1.0.0",
    "vendor", "takahasi",
    "category", "Category",
    "activity_type", "STATIC",
    "max_instance", "1",
    "language", "Python",
    "lang_type", "SCRIPT",
    "conf.default.model", "GoogLeNet_output2_3.model", # 変更
    "conf.default.labels", "face_cat_dog.txt", # 変更
    "conf.default.decision_rate", "0.3",
    "conf.default.decision_count", "2",
    "conf.default.display_num", "10",
    "conf.__widget__.model", "text",
    "conf.__widget__.labels", "text",
    "conf.__widget__.decision_rate", "slider",
    "conf.__widget__.decision_count", "slider",
    "conf.__widget__.display_num", "text",
    "conf.__type__.model", "string",
    "conf.__type__.labels", "string",
    "conf.__type__.decision_rate", "double",
    "conf.__type__.decision_count", "int",
    "conf.__type__.display_num", "int",
    "conf.__constraints__.decision_rate", "0.0<=x<=1.0",
    "conf.__constraints__.decision_count", "0<=x<=99",
    ""
]
# </rtc-template>


##
# @class ImageToObjectPrediction
# @brief predict image by chainer
#
#
class ImageToObjectPrediction(OpenRTM_aist.DataFlowComponentBase):

    ##
    # @brief constructor
    # @param manager Maneger Object
    #
    def __init__(self, manager):
        OpenRTM_aist.DataFlowComponentBase.__init__(self, manager)

        self._d_in_img = RTC.CameraImage(RTC.Time(0, 0), 0, 0, 0, [], 0, [])
        self._in_imageIn = OpenRTM_aist.InPort("in_image", self._d_in_img)

        self._d_out_img = RTC.CameraImage(RTC.Time(0, 0), 0, 0, 0, [], 0, [])
        self._out_imageOut = OpenRTM_aist.OutPort("out_image", self._d_out_img)

        self._d_out_name = RTC.TimedString(RTC.Time(0, 0), [])
        self._out_nameOut = OpenRTM_aist.OutPort("out_name", self._d_out_name)

        # initialize of configuration-data.
        # <rtc-template block="init_conf_param">
        """

         - Name:  model
         - DefaultValue: googlenet.model
        """
        self._model = ['GoogLeNet_output2_3.model'] # 変更
        """

         - Name:  labels
         - DefaultValue: labels.txt
        """
        self._labels = ['face_cat_dog.txt'] # 変更
        """

         - Name:  decision_rate
         - DefaultValue: 0.3
        """
        self._decision_rate = [0.3]
        """

         - Name:  decision_count
         - DefaultValue: 2
        """
        self._decision_count = [3]
        """

         - Name:  display_num
         - DefaultValue: 10
        """
        self._display_num = [10]

        # </rtc-template>

        self._net_model = network.GoogLeNet() # 変更
        self._log = OpenRTM_aist.Manager.instance().getLogbuf("ImageToObjectPrediction")
        self._previous_object = ""
        self._match_count = 0

    ##
    #
    # The initialize action (on CREATED->ALIVE transition)
    # formaer rtc_init_entry()
    #
    # @return RTC::ReturnCode_t
    #
    #
    def onInitialize(self):
        # Bind variables and configuration variable
        self.bindParameter("model", self._model, "GoogLeNet_output2_3.model") # 変更
        self.bindParameter("labels", self._labels, "face_cat_dog.txt") # 変更
        self.bindParameter("decision_rate", self._decision_rate, "0.3")
        self.bindParameter("decision_count", self._decision_count, "3")
        self.bindParameter("display_num", self._display_num, "10")

        # Set InPort buffers
        self.addInPort("in_image", self._in_imageIn)

        # Set OutPort buffers
        self.addOutPort("out_image", self._out_imageOut)
        self.addOutPort("out_name", self._out_nameOut)

        # Set service provider to Ports

        # Set service consumers to Ports

        # Set CORBA Service Ports

        return RTC.RTC_OK

    #   ##
    #   #
    #   # The finalize action (on ALIVE->END transition)
    #   # formaer rtc_exiting_entry()
    #   #
    #   # @return RTC::ReturnCode_t
    #
    #   #
    # def onFinalize(self):
    #
    #   return RTC.RTC_OK

    #   ##
    #   #
    #   # The startup action when ExecutionContext startup
    #   # former rtc_starting_entry()
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    # def onStartup(self, ec_id):
    #
    #   return RTC.RTC_OK

    #   ##
    #   #
    #   # The shutdown action when ExecutionContext stop
    #   # former rtc_stopping_entry()
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    # def onShutdown(self, ec_id):
    #
    #   return RTC.RTC_OK

        ##
        #
        # The activated action (Active state entry action)
        # former rtc_active_entry()
        #
        # @param ec_id target ExecutionContext Id
        #
        # @return RTC::ReturnCode_t
        #
        #
    def onActivated(self, ec_id):

        self._previous_object = ""
        self._match_count = 0
        serializers.load_npz(self._model[0], self._net_model, strict=False) # 変更

        return RTC.RTC_OK

        ##
        #
        # The deactivated action (Active state exit action)
        # former rtc_active_exit()
        #
        # @param ec_id target ExecutionContext Id
        #
        # @return RTC::ReturnCode_t
        #
        #
    def onDeactivated(self, ec_id):

        return RTC.RTC_OK

        ##
        #
        # The execution action that is invoked periodically
        # former rtc_active_do()
        #
        # @param ec_id target ExecutionContext Id
        #
        # @return RTC::ReturnCode_t
        #
        #
    def onExecute(self, ec_id):

        if not self._in_imageIn.isNew():
            return RTC.RTC_OK

        # convert image data
        data = self._in_imageIn.read()
        image = np.frombuffer(data.pixels, dtype=np.uint8)
        image = image.reshape(data.height, data.width, 3)
        pil_img = Image.fromarray(image)
        # RGB
        pil_img.save('data/save_pillow.jpg')

        # load image
        img = cv2.resize(image, (224, 224)).astype(np.float32)
        img = img.transpose(2, 0, 1).reshape(1, 3, 224, 224)

        # forward
        x = Variable(img)
        self._net_model.train = False
        y = self._net_model(x)

        # show prediction
        prediction = F.softmax(y)
        categories = np.loadtxt(self._labels[0], delimiter="\n", dtype=str)
        result = zip(prediction.data.reshape((prediction.data.size,)), categories)
        print("=================")
        result = sorted(result, reverse=True)
        print(result)
        for i, (score, label) in enumerate(result[:self._display_num[0]]):
            self._log.RTC_DEBUG('{:>3d} {:>6.2f}% {}'.format(i + 1, score * 100, label))

        if result[0][0] > float(self._decision_rate[0]):
            self._d_out_name.data = result[0][1]
            print("Is this " + str(self._d_out_name.data) + " ?")
            ans = input()
            if ans == "y":
                self._d_out_name.data = result[0][1]
                self._out_nameOut.write()
                self._log.RTC_INFO("Recognized Object: " + str(self._d_out_name.data))
            elif ans == "cat":
                self._d_out_name.data = "cat,0"
                self._out_nameOut.write()
                self._log.RTC_INFO("Recognized Object: " + str(self._d_out_name.data))
            elif ans == "normal":
                self._d_out_name.data = "normal,2"
                self._out_nameOut.write()
                self._log.RTC_INFO("Recognized Object: " + str(self._d_out_name.data))
            elif ans == "dog":
                self._d_out_name.data = "dog,1"
                self._out_nameOut.write()
                self._log.RTC_INFO("Recognized Object: " + str(self._d_out_name.data))
            else:
                self._d_out_name.data = "stop"
                self._out_nameOut.write()


            self._previous_object = result[0][1]

        return RTC.RTC_OK

    #   ##
    #   #
    #   # The aborting action when main logic error occurred.
    #   # former rtc_aborting_entry()
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    # def onAborting(self, ec_id):
    #
    #   return RTC.RTC_OK

    #   ##
    #   #
    #   # The error action in ERROR state
    #   # former rtc_error_do()
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    # def onError(self, ec_id):
    #
    #   return RTC.RTC_OK

    #   ##
    #   #
    #   # The reset action that is invoked resetting
    #   # This is same but different the former rtc_init_entry()
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    # def onReset(self, ec_id):
    #
    #   return RTC.RTC_OK

    #   ##
    #   #
    #   # The state update action that is invoked after onExecute() action
    #   # no corresponding operation exists in OpenRTm-aist-0.2.0
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #

    #   #
    # def onStateUpdate(self, ec_id):
    #
    #   return RTC.RTC_OK

    #   ##
    #   #
    #   # The action that is invoked when execution context's rate is changed
    #   # no corresponding operation exists in OpenRTm-aist-0.2.0
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    # def onRateChanged(self, ec_id):
    #
    #   return RTC.RTC_OK


def ImageToObjectPredictionInit(manager):
    profile = OpenRTM_aist.Properties(defaults_str=imagetoobjectprediction_spec)
    manager.registerFactory(profile,
                            ImageToObjectPrediction,
                            OpenRTM_aist.Delete)


def MyModuleInit(manager):
    ImageToObjectPredictionInit(manager)

    # Create a component
    manager.createComponent("ImageToObjectPrediction")


def main():
    mgr = OpenRTM_aist.Manager.init(sys.argv)
    mgr.setModuleInitProc(MyModuleInit)
    mgr.activateManager()
    mgr.runManager()

if __name__ == "__main__":
    main()
