Participants have to train a model for CT scan classification. This is a multi-way classification task in which an image must be classified into one of the three classes (native, arterial, venous).

The training data is composed of 15,000 image files. The validation set is composed of 4,500 image files. The test is composed of 3,900 image files.

**File Descriptions:**
train.txt - the training metadata file containing the training image file names and the corresponding labels (one example per row)
validation.txt - the validation metadata file containing the validation image file names and the corresponding labels (one example per row)
test.txt - the test metadata file containing the test image file names (one sample per row)
sample_submission.txt - a sample submission file in the correct format

**Metadata Files:**
The metadata files are provided in the following format based on comma separated values:

000001.png,0
000002.png,1
...

Each line represents an example where:
The first column shows the image file name of the example.
The second column is the label associated to the example.

**Image Files:**
The image files are provided in .png format.
