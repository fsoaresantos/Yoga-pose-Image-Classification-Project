# Yoga Pose API #

This API allows you to classify a image of a yoga pose by submiting an image file.

The API is available at `https://j4ddwyyv1k.execute-api.us-east-1.amazonaws.com/DEV/getyogaapp`

## Endpoints ##

### html page ###

GET `https://j4ddwyyv1k.execute-api.us-east-1.amazonaws.com/DEV/getyogaapp`

The response body will contain the API html page.

### Request a classification ###

POST `https://j4ddwyyv1k.execute-api.us-east-1.amazonaws.com/DEV/getyogaapp`

Allows you to submit an image for classification or request a classification of a random image from validation dataset.

The request body needs to be in JSON format and include the following properties:

 - `data` - string - optional
 - `ground_truth` - string - optional
 - `is_user_data` - string - "true" if a data is submited or empty for no data

Example: no data is submited - the user chose to use an image from the validation dataset
```
{
htmlMethod: POST
headers: {"Content-Type": "application/json"}

body: {
    "data": "",
    "ground_truth": "",
    "is_user_data": ""
}
}
```

The response body will be in JSON format and will contain the input image, the classifier prediction and the wikipedia page summary, if any, of the predicted value 

 - `data` - base64 encoded string
 - `groundtruth` - string
 - `prediction` - string
 - `title` - string
 - `summary` - string
 - `exception` - string

---

#### Possible errors ####

1) When trying to upload an invalid file type:

  > \`Please upload a valid image file OR click the Classify button to use one of our images\`

2) If predicted value does not match any wikipedia page :

  > `Issue: Page id "<predicted_value>" does not match any pages. Try another id!
  > Sorry, there is no wikipedia content to show here!`







