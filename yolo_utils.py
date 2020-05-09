import cv2
import numpy as np

def getOutputNames(net):
	layerNames = net.getLayerNames()

	return [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def postprocess_file(frame, outs, CONF_THRESH):
	frameHeight = frame.shape[0]
	frameWidth = frame.shape[1]	

	classIds = []
	confidences = []
	boxes = []

	for out in outs:
		for detection in out:
			scores = detection[5:]
			classId = np.argmax(scores)
			confidence = float(scores[classId])

			if confidence > CONF_THRESH:
				center_x = int(detection[0] * frameWidth)
				center_y = int(detection[1] * frameHeight)

				width = int(detection[2] * frameWidth)
				height = int(detection[3] * frameHeight)

				left = int(center_x - width / 2)
				top = int(center_y - height / 2)

				classIds.append(classId)
				confidences.append(confidence)
				boxes.append([left, top, width, height])

	return boxes, confidences, classIds			

def drawPredictions(frame, classes, classId, conf, left, top, right, bottom):
	cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

	label = '%.2f' % conf	

	if classes:
		assert(classId < len(classes))
		label = '%s : %s' % (classes[classId], label)

		labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
		top = max(top, labelSize[1])

		cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

