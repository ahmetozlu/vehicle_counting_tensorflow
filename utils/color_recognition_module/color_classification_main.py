#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 31st December 2017 - new year eve :)
#----------------------------------------------

import cv2
import color_histogram_feature_extraction
import knn_classifier
import os
import os.path

cap = cv2.VideoCapture(1)
ret, frame = cap.read()
prediction = "n.a."

#checking whether the training data is ready
PATH='./training.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print "training data is ready, classifier is loading..."
else:
    print "training data is being created..."
    open('training.data','w')
    color_histogram_feature_extraction.training()
    print "training data is ready, classifier is loading..."

'''while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    cv2.putText(frame, "Prediction: " + prediction, (15, 45), cv2.FONT_HERSHEY_PLAIN, 3, 200)

    # Display the resulting frame
    cv2.imshow('color classifier',frame)

    color_histogram_feature_extraction.color_histogram_of_test_image(frame)
    
    prediction = knn_classifier.main("training.data", "test.data")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()'''
