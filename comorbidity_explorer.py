import plotly.graph_objects as go
import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import webbrowser
from threading import Timer
import googlesearch

# transformer model stuff
from transformers import pipeline
from transformers import BertTokenizer
from transformers import BertForMaskedLM

# create the application and server
app = dash.Dash(__name__)
server = app.server

# read in raw data file for statistical analysis
raw_data = pd.read_csv("./data/conditions.csv")

# initialize the AI model for predictions
MODEL_FOLDER = "./synthBERT"
VOCAB_PATH = "./synthBERT/vocab.txt"

# load vocab file to list
vocab_file = open(VOCAB_PATH, "r")
vocab = vocab_file.read().split("\n")

# initialize model variables
tokenizer = None
model = None
fill_mask = None

# load conditions for autofill suggestions
suggestions = raw_data['DESCRIPTION'].tolist()
suggestions = list(dict.fromkeys(suggestions))


# ------------------------- setup an initial figure to display while the page loads -------------------------
pageLoadingData = pd.DataFrame({"PREDICTION":["Loading", "Please wait..."], "CONFIDENCE":[1.5, 8.0]})
pageLoadingText = ["Loading................................................", html.Br(), "Please wait...................................."]
def generate_loading_graph(data):
    try: 
        data.size
    except:
        return []
    fig = px.bar(data, x='CONFIDENCE', y='PREDICTION',
                 hover_data=['PREDICTION', 'CONFIDENCE'],
                 labels={'PREDICTION':'predicted condition', 'CONFIDENCE':'confidence', 'DESCRIPTION':''},
                color = 'CONFIDENCE',
                height = 400,
                width = 600,
                orientation = 'h')
    fig.update_layout(
                height=400,
                showlegend=False,
                margin=dict(
                    l=10,
                    r=50,
                    b=20,
                    t=20,
                    pad=4
                ))
    fig.layout.paper_bgcolor = '#fff5bf'
    fig.update_yaxes(visible=False, showticklabels=False)
    return fig
pageLoadingFig = generate_loading_graph(pageLoadingData)
# ------------------------- ------------------------------------------------------ -------------------------


# ----------------------------------------- setup hover info text ------------------------------------------
tooltip = html.Div(
    [
        html.Div(id = "tooltip-target", children =
            [
                html.Img(id = "info-button", src = app.get_asset_url('./images/info_icon_purple.png'), alt = 'purple information button', style={'width': '15%', 'height': '15%', 'marginTop':'20px'})
            ]
        ),
        dbc.Tooltip(
            ["Use the text fields above to add/remove medical conditions from your search and prediction.", html.Br(),"The top graph uses an AI language model to predict the next most-likely condition, based on a patient dataset.", html.Br(),"Chronological inputs will yield more accurate results, since the model was trained on a chronological dataset.", html.Br(),"Click any of the predicted conditions for more info about them.", html.Br(),"All comorbidities of patients in the dataset with those conditions will be graphed below.", html.Br(),"Mouseover any bar in the chart for more info."],
            target="tooltip-target",
            hide_arrow=False,
            placement="left",
            style={'backgroundColor': 'black', 'color':'white', "font-family":"Arial Black", "fontSize":"16px"}
        ),
        html.P(["HOW DOES", html.Br(), "THIS WORK?"], style={"float":"left", "marginLeft":"50px", "text-align":"center", "font-family":"IntroCondBlackFree", "fontSize":"22px"})
    ]
)
# ----------------------------------------- -------------------- ------------------------------------------


# --------------------------- layout the webpage ------------------------
app.layout = html.Div(children = 
                        [html.Div("Comorbidity Explorer", style={
                                                "color":"black",
                                                "text-align":"center",
                                                "font-family":"IntroCondBlackFree",
                                                "fontSize":"44px",
                                                "marginTop":"10px"
                                                }),
                        html.Img(id = "past-present-timeline", src = app.get_asset_url('./images/past-present-bar.png'), alt = 'past to present timeline', style={'width':'85%', 'marginTop':'20px', 'marginLeft':'100px', 'marginRight':'50px'}),
                        html.Span(["PAST", html.Div(style={"width":"80%", "height":"18px", "backgroundColor":"#fff5bf'", "display":"inline-block"}), "PRESENT"], style={"fontSize":"18px", "font-family":"IntroCondLightFree", "marginBottom":"20px", "marginLeft":"80px", "clear":"both"}),
                        html.Datalist(id='list-suggested-inputs', children=[html.Option(value=word) for word in suggestions]),
                        html.Div(dcc.Input(id='input-1',type='text',list='list-suggested-inputs',value='Viral sinusitis (disorder)', placeholder='Condition 1', debounce=True, style={'fontSize':'18px', 'width': '100%'}), style={'width': '15%', 'height':'30px', 'display': 'inline-block', 'marginTop': '-20px', 'marginLeft': '30px', 'fontSize':'18px'}),
                        html.Img(id = "time-arrow-1", src = app.get_asset_url('./images/time-arrow-1.png'), alt = 'timeline indicator 1', style={'vertical-align':'middle', 'marginLeft':'10px', 'marginRight':'10px'}),
                        html.Div(dcc.Input(id='input-2',type='text',list='list-suggested-inputs',value='', placeholder='Condition 2', debounce=True, style={'fontSize':'18px', 'width': '100%'}), style={'width': '15%', 'height':'30px', 'display': 'inline-block', 'marginTop': '-20px', 'marginLeft': '0px', 'fontSize':'18px'}),
                        html.Img(id = "time-arrow-2", src = app.get_asset_url('./images/time-arrow-2.png'), alt = 'timeline indicator 2', style={'vertical-align':'middle', 'marginLeft':'10px', 'marginRight':'10px'}),
                        html.Div(dcc.Input(id='input-3',type='text',list='list-suggested-inputs',value='', placeholder='Condition 3', debounce=True, style={'fontSize':'18px', 'width': '100%'}), style={'width': '15%', 'height':'30px', 'display': 'inline-block', 'marginTop': '-20px', 'marginLeft': '0px', 'fontSize':'18px'}),
                        html.Img(id = "time-arrow-3", src = app.get_asset_url('./images/time-arrow-3.png'), alt = 'timeline indicator 3', style={'vertical-align':'middle', 'marginLeft':'10px', 'marginRight':'10px'}),
                        html.Div(dcc.Input(id='input-4',type='text',list='list-suggested-inputs',value='', placeholder='Condition 4', debounce=True, style={'fontSize':'18px', 'width': '100%'}), style={'width': '15%', 'height':'30px', 'display': 'inline-block', 'marginTop': '-20px', 'marginLeft': '0px', 'fontSize':'18px'}),
                        html.Img(id = "time-arrow-4", src = app.get_asset_url('./images/time-arrow-4.png'), alt = 'timeline indicator 4', style={'vertical-align':'middle', 'marginLeft':'10px', 'marginRight':'10px'}),
                        html.Div(dcc.Input(id='input-5',type='text',list='list-suggested-inputs',value='', placeholder='Condition 5', debounce=True, style={'fontSize':'18px', 'width': '100%'}), style={'width': '15%', 'height':'30px', 'display': 'inline-block', 'marginTop': '-20px', 'marginLeft': '0px', 'fontSize':'18px'}),
                        html.Div(html.Button("Search", id = "searchButton")),
                        html.Div(id='predictedText', children = pageLoadingText, style={"font-family":"Arial Black",
                                                                            "fontSize":"22px",
                                                                            "width":"auto",
                                                                            "max-width":"500px",
                                                                            "float": "left",
                                                                            "text-align": "right"
                                                                            }),
                        dcc.Loading(id="predictionGraphLoading", children=[dcc.Graph(id = 'predictedGraph', figure = pageLoadingFig, style={"float": "left",
                                                                                                                                            "max-height": "400px"})
                                                                            ], type = "cube", color="#7a41a3"
                                ),
                        tooltip,
                        html.Div(id='direct-patients', children = "", style={"font-family":"Arial Black",
                                                                            "fontSize":"18px", 
                                                                            "color":"red",
                                                                            "clear":"both" }),
                        dcc.Loading(id="dataGraphLoading", children=[dcc.Graph(id = 'myGraph', figure = [])], type="default"),
                        html.Div("condition", style={'text-align':'center', 'font-family':'Verdana'})
                      ], style={'backgroundColor':'#fff5bf'})
# ------------------------------------------------------------------------


### callback buttons ###
@app.callback(
    Output('myGraph', 'figure'),
    [Input('input-1', 'value'), Input('input-2', 'value'), Input('input-3', 'value'), Input('input-4', 'value'), Input('input-5', 'value')]
)

def update_my_graph(input_condition1, input_condition2, input_condition3, input_condition4, input_condition5):
    input_cond1 = None
    input_cond2 = None
    input_cond3 = None
    input_cond4 = None
    input_cond5 = None
    if len(input_condition1) > 0:
        input_cond1 = input_condition1.strip()
    if len(input_condition2) > 0:
        input_cond2 = input_condition2.strip()
    if len(input_condition3) > 0:
        input_cond3 = input_condition3.strip()
    if len(input_condition4) > 0:
        input_cond4 = input_condition4.strip()
    if len(input_condition5) > 0:
        input_cond5 = input_condition5.strip()
        
    reduced_conditions = reduce_conditions(input_condition1, input_cond2, input_cond3, input_cond4, input_cond5)
    data, adj_patients = get_stats(raw_data, reduced_conditions)
    fig = generate_my_graph(data)
    return fig


@app.callback(
    Output('direct-patients', 'children'),
    [Input('input-1', 'value'), Input('input-2', 'value'), Input('input-3', 'value'), Input('input-4', 'value'), Input('input-5', 'value')]
)

def update_patient_counter(input_condition1, input_condition2, input_condition3, input_condition4, input_condition5):
    input_cond1 = None
    input_cond2 = None
    input_cond3 = None
    input_cond4 = None
    input_cond5 = None
    if len(input_condition1) > 0:
        input_cond1 = input_condition1.strip()
    if len(input_condition2) > 0:
        input_cond2 = input_condition2.strip()
    if len(input_condition3) > 0:
        input_cond3 = input_condition3.strip()
    if len(input_condition4) > 0:
        input_cond4 = input_condition4.strip()
    if len(input_condition5) > 0:
        input_cond5 = input_condition5.strip()
        
    reduced_conditions = reduce_conditions(input_condition1, input_cond2, input_cond3, input_cond4, input_cond5)
    data, adj_patients = get_stats(raw_data, reduced_conditions)
    return "Patients in database with listed conditions: " + str(adj_patients)


@app.callback(
    Output('predictedGraph', 'figure'),
    Output('predictedText', 'children'),
    [Input('input-1', 'value'), Input('input-2', 'value'), Input('input-3', 'value'), Input('input-4', 'value'), Input('input-5', 'value')]
)

def update_prediction_graph(input_condition1, input_condition2, input_condition3, input_condition4, input_condition5):
    input_cond1 = ""
    input_cond2 = ""
    input_cond3 = ""
    input_cond4 = ""
    input_cond5 = ""
    if len(input_condition1) > 0:
        input_cond1 = input_condition1.strip().replace(" ", "_")
    if len(input_condition2) > 0:
        input_cond2 = input_condition2.strip().replace(" ", "_")
    if len(input_condition3) > 0:
        input_cond3 = input_condition3.strip().replace(" ", "_")
    if len(input_condition4) > 0:
        input_cond4 = input_condition4.strip().replace(" ", "_")
    if len(input_condition5) > 0:
        input_cond5 = input_condition5.strip().replace(" ", "_")

    inputs = [input_cond1, input_cond2, input_cond3, input_cond4, input_cond5]

    # only setup the model after page loads
    global tokenizer
    global model
    global fill_mask

    # only initialize the model once 
    if(tokenizer == None):
        # load the pretrained model
        tokenizer = BertTokenizer(VOCAB_PATH, do_basic_tokenize=True, additional_special_tokens=vocab)
        model = BertForMaskedLM.from_pretrained(MODEL_FOLDER)

        # setup the prediction pipeline
        fill_mask = pipeline(
            "fill-mask",
            model= model,
            tokenizer=tokenizer,
            topk = 10
        )

    # set up the string to run prediction on
    input_prediction_string = ""
    for input_cond in inputs:
        if len(input_cond) > 0:
            input_prediction_string += input_cond + " "

    if input_prediction_string == "":
        # no conditions input, return blank figure
        return ""
    else:

        input_prediction_string += "[MASK]"

        # run the preditions on the inputs ("condition_1 condition_2 [MASK]")
        predictions = fill_mask(input_prediction_string)

        split_predictions = []
        for prediction in predictions:
            split_predictions.append([prediction['token_str'],  str(prediction['score']*100)])

        # post-process the predictions
        predicted_conditions = []
        predicted_confidences = []
        for condition in split_predictions:
            predicted_conditions.append(condition[0].replace("_", " "))
            predicted_confidences.append(condition[1])

        data = pd.DataFrame({"PREDICTION":predicted_conditions, "CONFIDENCE":predicted_confidences})
        data= data.sort_values(by=['CONFIDENCE'], ascending=True) 

        # make the predicitons figure
        fig = generate_prediction_graph(data)
        
        # make the links that appear next to the prediction figure
        output_prediction_html = []
        for condition in predicted_conditions:
            reference_link = googlesearch.search("Mayo clinic " + condition, num_results=1)[0]
            output_prediction_html.append(html.A(condition, href=reference_link, target="_blank"))
            output_prediction_html.append(html.Br())

        return fig, html.P(output_prediction_html)

###         ###


# input validation
def reduce_conditions(condition_1, condition_2 = None, condition_3 = None, condition_4 = None, condition_5 = None):
    
    submitted_conditions = [condition_1, condition_2, condition_3, condition_4, condition_5]
    reduced_conditions = []
    
    for condition in submitted_conditions:
        if condition != None:
            if condition not in reduced_conditions:
                reduced_conditions.append(condition)
                
    return reduced_conditions


# reduce the dataset using a single condition to find adjacent patients
def reduce_df(dataframe, condition):
    
    condition_matched_df = dataframe[dataframe['DESCRIPTION'] == condition]
    if condition_matched_df.size == 0:
        return None, 0
    
    patient_list = condition_matched_df['PATIENT'].tolist()
    adjacent_patients = len(patient_list)
    patient_matched_df = dataframe[dataframe['PATIENT'].isin(patient_list)]
    reduced_df = patient_matched_df[patient_matched_df['DESCRIPTION'] != condition]
    reduced_df = reduced_df.drop_duplicates(subset=['PATIENT', 'DESCRIPTION'])
    return reduced_df, adjacent_patients


# recursively reduce the dataset using each input condition to find similar patients
def fully_reduce_df(dataframe, condition_list):
    
    reduced_df = dataframe
    adj_patients = 0
    
    for condition in condition_list:
        reduced_df, adj_patients = reduce_df(reduced_df, condition)
        if adj_patients == 0:
            return None, 0
    
    return reduced_df, adj_patients


# format reduced dataframe into number and percentage of patients with each condition
def get_stats(dataframe, condition_list):
    
    reduced_df, adj_patients = fully_reduce_df(dataframe, condition_list)
    
    if adj_patients == 0:
        return {}, 0
    
    adj_conditions = reduced_df['DESCRIPTION'].tolist()
    adj_conditions_list = list(dict.fromkeys(adj_conditions))
    
    counts = dict()
    for condition in adj_conditions:
        counts[condition] = counts.get(condition, 0) + 1
    
    stats = dict()
    for condition in adj_conditions_list:
        stats[condition] = round((float(counts[condition])/float(adj_patients)) * 100.0, 2)
    
    counts_df = pd.DataFrame.from_dict([counts])
    stats_df = pd.DataFrame.from_dict([stats])
    frames = [counts_df, stats_df]
    data_df = pd.concat(frames)
    data_df = data_df.transpose()
    data_df = data_df.reset_index()
    data_df.columns = ['DESCRIPTION', 'COUNT', 'PERCENT']
    data_df = data_df.sort_values(by=['COUNT'], ascending=False)
        
    return  data_df, adj_patients


# make the lower graph of all patients
def generate_my_graph(data):
    try: 
        data.size
    except:
        return []
    fig = px.bar(data, x='DESCRIPTION', y='PERCENT',
                 hover_data=['COUNT', 'PERCENT'],
                 labels={'COUNT':'count', 'PERCENT':'percentage', 'DESCRIPTION':''},
                color = 'PERCENT', range_color=[0,100],
                height = 600)
    fig.update_layout(
                height=600,
                margin=dict(
                    l=50,
                    r=50,
                    b=200,
                    t=20,
                    pad=4
                ))
    fig.update_xaxes(ticklen=10) 
    #fig.layout.plot_bgcolor = '#fff5bf'
    fig.layout.paper_bgcolor = '#fff5bf'
    return fig


# make the prediction graph
def generate_prediction_graph(data):
    try: 
        data.size
    except:
        return []
    fig = px.bar(data, x='CONFIDENCE', y='PREDICTION',
                 hover_data=['PREDICTION', 'CONFIDENCE'],
                 labels={'PREDICTION':'predicted condition', 'CONFIDENCE':'confidence (%)', 'DESCRIPTION':''},
                color = 'CONFIDENCE',
                height = 400,
                width = 600,
                orientation = 'h')
    fig.update_layout(
                height=400,
                showlegend=False,
                margin=dict(
                    l=10,
                    r=50,
                    b=20,
                    t=20,
                    pad=4
                ))
    fig.layout.paper_bgcolor = '#fff5bf'
    fig.update_yaxes(visible=False, showticklabels=False, categoryorder='total ascending')
    return fig


def open_browser(url):
    webbrowser.open_new(url)


if __name__ == '__main__':
    # Uncomment the next line if running locally - this will automatically open the app page
    # Timer(1, open_browser, args=('http://127.0.0.1:8050',)).start()
    app.run_server()
