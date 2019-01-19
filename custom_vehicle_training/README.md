# VEHICLE DETECTION, TRACKING AND COUNTING
This sample project focuses on "Vechicle Detection, Tracking and Counting" using [**TensorFlow Object Counting API**](https://github.com/ahmetozlu/tensorflow_object_counting_api). 

### under construction

---

The vehicle detection&counting challenge model will be published in here! **The training is on progress**!

Some traffic jam videos:

- https://www.youtube.com/watch?v=ynyImulYA8M

- https://www.youtube.com/watch?v=emDtxVZqZTA

---

- Run the command in "tensorflow/models/research/" to solve the "object_detection module can not found" issue:

    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

utils:

- generate_tfrecord.py: to convert our XML files to TFRecords

- replace_text_in_files.py: to change specific strings in annotation xml such as changing <path>

- xml_to_csv.py: to convert xml to csv so we can generate TFRecords from xml files

