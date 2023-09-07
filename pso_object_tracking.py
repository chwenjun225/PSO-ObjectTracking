import numpy as np
import cv2
import matplotlib.pyplot as plt

if "pso-object-tracking":
    def RescaleImage(image, scale=0.5):
        return cv2.resize(image,
                          (int(image.shape[1] * (1 - scale)), int(image.shape[0] * (1 - scale))),
                          interpolation=cv2.INTER_AREA)


    def findIndexInArrays(arrays, value):
        for i in range(len(arrays)):
            list_numbers = list(arrays[i])
            element = value[i]
            index_ = list_numbers.index(element)
            print("Index of {} is: {}".format(element, index_))


    def Calculate_pBest_Values(object_, object_origin):
        hue_obj = object_[0]
        gray_obj = object_[1]
        hue_obj_origin = object_origin[0]
        gray_obj_origin = object_origin[1]
        hue_distribution = cv2.compareHist(hue_obj, hue_obj_origin, cv2.HISTCMP_BHATTACHARYYA)
        gray_distribution = cv2.compareHist(gray_obj, gray_obj_origin, cv2.HISTCMP_BHATTACHARYYA)
        return [hue_distribution, gray_distribution]


    def Calculate_Multi_pBest_Values(multiObjectFeatures):
        global object_originals
        return [Calculate_pBest_Values(multiObjectFeatures[i], object_originals[i]) for i in range(len(multiObjectFeatures))]


    def CutObjectFromFrame(frame):
        x, y, w, h = cv2.selectROI("Select Object", frame)
        return frame[y:y + h, x:x + w]


    def DivideObjectIntoFourPart(image):
        w, h = image.shape[0:2]
        return [
            image[0:w // 2, 0:h // 2],
            image[w // 2:w, 0:h // 2],
            image[w // 2: w, h // 2:h],
            image[0:w // 2, h // 2:h],
        ]


    def ProcessingImage(image):
        imageHue = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 0]
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return [imageHue, imageGray]


    def ProcessingMultiPartImage(images):
        return [ProcessingImage(image) for image in images]


    def FeatureOfEachPart(imageProcessed):
        imageHue = imageProcessed[0]
        imageGray = imageProcessed[1]
        histHue = cv2.calcHist([imageHue], [0], None, [180], [0, 180])
        histGray = cv2.calcHist([imageGray], [0], None, [180], [0, 255])
        histHueNorm = cv2.normalize(histHue, histHue, 0, 255, cv2.NORM_MINMAX)
        histGrayNorm = cv2.normalize(histGray, histGray, 0, 255, cv2.NORM_MINMAX)
        return [histHueNorm, histGrayNorm]


    def FeatureOfMultiParts(multiImageProcessed):
        return [FeatureOfEachPart(imageProcessed) for imageProcessed in multiImageProcessed]


    def ObjectFeatures(objectFromFrame):
        objectSplitFourParts = DivideObjectIntoFourPart(objectFromFrame)
        objectProcessed = [ProcessingImage(obj) for obj in objectSplitFourParts]
        featuresObjectFourPart = [FeatureOfEachPart(objProc) for objProc in objectProcessed]
        return featuresObjectFourPart


    def ConvertToCOCO(stateOfVector):
        """
        Convert "coordinates" from stateOfVector to COCO standard.
        :param stateOfVector: List(), Tuple()
        :return: Tuple()
        """
        x, y, w, h = stateOfVector
        x1 = x - w // 2
        y1 = y - h // 2
        x2 = x + w // 2
        y2 = y + h // 2
        return np.array([x1, y1, x2, y2])


    def calc_sum(pBest_value):
        return sum([sum(pBes) for pBes in pBest_value])


    def Get_pBest_Objectives(frame, stateOfVectors_):
        x1, y1, x2, y2 = ConvertToCOCO(stateOfVectors_)
        objects_ = [frame[int(y1[i]):int(y2[i]), int(x1[i]):int(x2[i])] for i in range(len(x1))]
        fourPartObjects = [DivideObjectIntoFourPart(obj_) for obj_ in objects_]
        objects_processed = [ProcessingMultiPartImage(fourPartObject) for fourPartObject in fourPartObjects]
        multi_objects_features = [FeatureOfMultiParts(obj_proc) for obj_proc in objects_processed]
        pBest_values = [Calculate_Multi_pBest_Values(object_features) for object_features in multi_objects_features]
        pBest_objs_ = [calc_sum(pBest_value) for pBest_value in pBest_values]
        return np.array(pBest_objs_)


    def UpdateToFind_gBest(
            frame_param,
            velocities_param,  # V
            pBest_param,  # pbest
            pBest_objectives_param,  # pbest_obj
            gBest_param,  # gbest
            gBest_objectives_param,  # gbest_obj
            stateOfVectors_param,  # X
            c1=0.5, c2=0.5, inertial_weight=0.8):

        image = frame_param
        V = velocities_param
        X = stateOfVectors_param
        pbest = pBest_param
        pbest_obj = pBest_objectives_param
        gbest = gBest_param
        gbest_obj = gBest_objectives_param
        w = inertial_weight

        r1, r2 = np.random.rand(2)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest.reshape(-1, 1) - X)
        X = X + V
        obj = Get_pBest_Objectives(image, X)
        pbest[:, (pbest_obj <= obj)] = X[:, (pbest_obj <= obj)]
        pbest_obj = np.array([pbest_obj, obj]).max(axis=0)
        gbest = pbest[:, pbest_obj.argmax()]
        gbest_obj = max(pbest_obj)
        print("gbest_obj: ", gbest_obj)
        return gbest


    def PSO_Object_Tracking(particles=200, objectFromFrame=None):
        global object_originals, stateOfVectors, gBest

        stateOfVectors = np.array([0, 0, 0, 0], dtype=object)
        gBest = np.array([0, 0, 0, 0], dtype=object)

        path = cv2.VideoCapture("E:/Study/PSO_Object_Tracking/test1.mp4") # Edit this path to test 
        # path = cv2.VideoCapture("E:/Study/PSO_Object_Tracking/test2.mp4")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("Can not open Camera...")
            exit()

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)
        result = cv2.VideoWriter('pso-object-tracking-result.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

        while True:
            ret, frame = cap.read()

            h_frame, w_frame = frame.shape[0:2]

            if not ret:
                print("Can not receive frame (stream end?). Exiting...")
                break

            if (objectFromFrame is None) and (sum(gBest) == 0):
                objectFromFrame = CutObjectFromFrame(frame)
                object_originals = ObjectFeatures(objectFromFrame)

                h_object, w_object = objectFromFrame.shape[0:2]
                x = np.random.randint(
                    w_object // 2,
                    w_frame - w_object // 2,
                    size=particles)
                y = np.random.randint(
                    h_object // 2,
                    h_frame - h_object // 2,
                    size=particles)
                w = np.array([w_object] * particles)
                h = np.array([h_object] * particles)

                stateOfVectors = np.array([x, y, w, h], dtype=object)
                velocities = np.random.randn(4, particles)
                pBest = stateOfVectors
                pBest_objectives = Get_pBest_Objectives(
                    frame,
                    stateOfVectors)
                gBest = pBest[:, pBest_objectives.argmax()]
                gBest_objective = max(pBest_objectives)

            gBest = UpdateToFind_gBest(
                frame_param=frame,
                velocities_param=velocities,
                pBest_param=pBest,
                pBest_objectives_param=pBest_objectives,
                gBest_param=gBest,
                gBest_objectives_param=gBest_objective,
                stateOfVectors_param=stateOfVectors,
                c1=0.5, c2=0.5, inertial_weight=0.9
            )
            print("gbest: ", gBest)
            x1_draw, y1_draw, x2_draw, y2_draw = ConvertToCOCO(gBest)

            cv2.rectangle(frame, (int(x1_draw), int(y1_draw)), (int(x2_draw), int(y2_draw)), (0, 0, 255), 3)
            result.write(frame)
            cv2.imshow("objectFromFrame", frame)
            if cv2.waitKey(25) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    global object_originals, stateOfVectors, gBest
    PSO_Object_Tracking()

