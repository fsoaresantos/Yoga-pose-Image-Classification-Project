import json
import base64
import random
import numpy as np  ## layer
import wikipedia as wiki  ## layer
import boto3
import sagemaker  ## layer
from sagemaker.serializers import IdentitySerializer
from sagemaker.predictor import Predictor

sm_session = sagemaker.Session()

s3_client = boto3.client("s3")
s3_resource = boto3.resource("s3")

# run a prediction on the endpoint using Boto3
runtime = boto3.Session().client(service_name="runtime.sagemaker") #"sagemaker-runtime")

endpoint_Name="pytorch-inference-2022-11-16-09-51-07-598" #endpoint name here

bucket = "capstone-project-sett22" #"capstone-project-yoga-pose-app"
prefix = "yoga-data"

val_data = f"{prefix}/val"
user_data = f"{prefix}/upload"

image_categories = ['adho mukha svanasana','adho mukha vriksasana','agnistambhasana','ananda balasana',
'anantasana','anjaneyasana','ardha bhekasana','ardha chandrasana','ardha matsyendrasana','ardha pincha mayurasana',
'ardha uttanasana','ashtanga namaskara','astavakrasana','baddha konasana','bakasana','balasana','bhairavasana',
'bharadvajasana i','bhekasana','bhujangasana','bhujapidasana','bitilasana','camatkarasana','chakravakasana',
'chaturanga dandasana','dandasana','dhanurasana','durvasasana','dwi pada viparita dandasana','eka pada koundinyanasana i',
'eka pada koundinyanasana ii','eka pada rajakapotasana','eka pada rajakapotasana ii','ganda bherundasana',
'garbha pindasana','garudasana','gomukhasana','halasana','hanumanasana','janu sirsasana','kapotasana','krounchasana',
'kurmasana','lolasana','makara adho mukha svanasana','makarasana','malasana','marichyasana i','marichyasana iii',
'marjaryasana','matsyasana','mayurasana','natarajasana','padangusthasana','padmasana','parighasana','paripurna navasana',
'parivrtta janu sirsasana','parivrtta parsvakonasana','parivrtta trikonasana','parsva bakasana','parsvottanasana',
'pasasana','paschimottanasana','phalakasana','pincha mayurasana','prasarita padottanasana','purvottanasana',
'salabhasana','salamba bhujangasana','salamba sarvangasana','salamba sirsasana','savasana','setu bandha sarvangasana',
'simhasana','sukhasana','supta baddha konasana','supta matsyendrasana','supta padangusthasana','supta virasana',
'tadasana','tittibhasana','tolasana','tulasana','upavistha konasana','urdhva dhanurasana','urdhva hastasana',
'urdhva mukha svanasana','urdhva prasarita eka padasana','ustrasana','utkatasana','uttana shishosana','uttanasana',
'utthita ashwa sanchalanasana','utthita hasta padangustasana','utthita parsvakonasana','utthita trikonasana',
'vajrasana','vasisthasana','viparita karani','virabhadrasana i','virabhadrasana ii','virabhadrasana iii',
'virasana','vriksasana','vrischikasana','yoganidrasana']



def grab_val_data():
    """randomly pick an image from validation data to use as input data"""

    bckt = s3_resource.Bucket(bucket)

    # Randomly pick a class in image_categories 
    label = random.choice(image_categories)
    # Fetch all files within the chosen validation folder
    pfx = f"{val_data}/{label}"  #e.g. yoga-data/val/virabhadrasana i
    objects = bckt.objects.filter(Prefix = pfx)
    # Grab a random object key (s3 'path') from the validation folder
    key = random.choice([x.key for x in objects]) #e.g. 'yoga-data/val/virabhadrasana i/20-0.jpg'
    
    """
    # Save object to local path (download from s3)
    selected_obj = f"s3://{bucket}/{key}"
    #local_obj_path = f"temp/{selected_obj.split('val/')[1].split('/')[1]}" ###
    local_obj_path = f"{selected_obj.split('val/')[1].split('/')[1]}" ###
    s3_resource.meta.client.download_file(Bucket = bucket, Key = key, Filename = local_obj_path) ###

    # Open object into a variable
    with open(local_obj_path, 'rb') as f:
        payload = f.read()
    """
    
    file = s3_client.get_object(Bucket=bucket, Key=key)
    payload = file['Body'].read()
    
    # Grab object ground truth from its s3 path
    groundtruth = key.split('val/')[1].split('/')[0]

    return(payload, groundtruth)



def call_endpoint(input_data):
    """query the Endpoint"""

    #serveless inference
    inferences = runtime.invoke_endpoint(
        EndpointName = endpoint_Name,
        ContentType= 'image/jpeg',
        #Accept='image/jpeg', ####
        Body=input_data
    )

    #fetch inferences (array of predictions)
    output = inferences['Body'].read().decode('utf-8')
    
    # convert result to ndarray
    output_array = json.loads(output)
    
    # find the class with maximum probability (prediction)
    prediction = image_categories[np.argmax(output_array)]
    print(f"Prediction: {prediction}") ###

    return(output_array, prediction)



def wiki_info(prediction):
    """grab wikipedia information (if any)"""

    exception = ''
    # If any information in wikipedia, pass, else, rise exception
    try:
        #fetch the wikipedia article that best matches the prediction
        article = wiki.page(prediction)
        summary = article.summary
        title = article.title
    except Exception as e:
        exception = f"issue: {e}"
        summary = ''
        title = prediction

    return(title, summary, exception)



def lauch_classifier(is_user_data, data = "", gt = ""):
    """
    args:
        is_user_data: 'true', ##type string, true if user have uploaded a file, empty otherwise
        data: b'', ##type bytes (base64 encoded image)
        gt: type string
    """

    if(is_user_data):
        ## add 'b' at begining of image string
        temp_image = data.encode('ascii')
        ## decode data to send to endpoint
        input_image = base64.b64decode(temp_image)
        
        ##call endpoint on user data (image)
        inferences, prediction = call_endpoint(input_image)
        
        ##grab wiki info
        title, summary, exception = wiki_info(prediction)
        
        ##return prediction (and groundtruth?)
        return {
            'image_data': base64.b64encode(input_image).decode('ascii'),
            'ground_truth': gt,
            'prediction': prediction,
            'title': title,
            'summary': summary,
            'exception': exception,
        }
    else:
        ##random select validation data from s3 bucket/val
        input_image, gt = grab_val_data()
        
        ##call endpoint on s3 random image data
        inferences, prediction = call_endpoint(input_image)
        
        ##grab wiki info
        title, summary, exception = wiki_info(prediction)
        
        #encode image data
        img_encoded = base64.b64encode(input_image)
        # convert bytes image to string (remove 'b')
        img_encoded_str = img_encoded.decode('ascii')

        ##return prediction, groundtruth, and image file
        return {
            'image_data': img_encoded_str,
            'ground_truth': gt,
            'prediction': prediction,
            'title': title,
            'summary': summary,
            'exception': exception,
        }



def lambda_handler(event, context):
    """
    test code:
    {
        "httpMethod": "GET", #"POST"
        "body": {
            "data": "b'xxx'", ##type bytes (base64 encoded image) #OR None
            "ground_truth": "", ##type string (given by the user)
            "is_user_data": "True", #False ##type bool
        }
    }

    Logic:
    if(event["httpMethod"] == "GET"): render html content.
    if(event["httpMethod"] == "POST"): takes webapp information and run the classifier,
    then, changes html content accordingly to display lauch_classifier() output to webapp.
    """

    if event["httpMethod"] == "GET":
        htmlFile = open("starter.html", "r")
        htmlContent = htmlFile.read()

        ##render initial website
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "text/html"},
            "body": htmlContent
        }

    if event["httpMethod"] == "POST":
        input_data = json.loads(event["body"])
        
        ##call endpoint
        guessing = lauch_classifier(input_data["is_user_data"], input_data["data"], input_data["ground_truth"])
        
        ## oss guessing["image_data"] was base64 encoded in lauch_classifier()
        output_data = {
            "data": guessing["image_data"],
            "groundtruth": guessing["ground_truth"],
            "prediction": guessing["prediction"],
            "title": guessing["title"],
            "summary": guessing["summary"],
            "exception": guessing["exception"]
        }

        ##update webpage by returning model output
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(output_data)
        }

