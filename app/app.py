
import flask
from flask import Flask, request, render_template
import json
from pythonosc import osc_server, dispatcher, udp_client
from threading import Thread
import main

app = Flask(__name__)

OSC_RECEIVE_IP = "127.0.0.1"
OSC_RECEIVE_PORT = 8001
OSC_SEND_PORT = 5001

osc_client = udp_client.SimpleUDPClient(OSC_RECEIVE_IP, OSC_SEND_PORT)

def handle_osc_input(address, *args):
    # Get the received OSC word
    input_text = args[0]
    print(f"Received OSC message: {input_text}")

    # Process the word using the predictive engine
    try:
        # Add a mask token (or any other necessary formatting for your model)
        input_text_with_mask = input_text + " <mask>"
        print(f"Input with mask: {input_text_with_mask}")

        # Use the input word in the predictive engine
        top_k = 5  # Top K predictions, you can adjust this
        predictions = main.get_all_predictions(input_text_with_mask, top_clean=top_k)
        print(f"Predictions: {predictions}")

        if predictions and 'bart' in predictions:
            # Get the predicted word (first one in the list)
            predicted_word = predictions['bart'].split('\n')[0]  # Take the first prediction
            print(f"Predicted word: {predicted_word}")

            # Send the prediction back to TouchDesigner via OSC
            osc_client.send_message("/predicted_word", predicted_word)
            print(f"Sent OSC message: {predicted_word}")
    except Exception as e:
        print(f"Error handling OSC input: {e}")


def start_osc_server():
    dispatcher_map = dispatcher.Dispatcher()
    dispatcher_map.map("/input_text", handle_osc_input)  # Path for receiving OSC
    server = osc_server.ThreadingOSCUDPServer((OSC_RECEIVE_IP, OSC_RECEIVE_PORT), dispatcher_map)
    print(f"Serving OSC on {OSC_RECEIVE_IP}:{OSC_RECEIVE_PORT}")
    server.serve_forever()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_end_predictions', methods=['POST'])
def get_prediction_eos():
    try:
        input_text = ' '.join(request.json['input_text'].split())
        input_text += ' <mask>'
        top_k = request.json['top_k']
        res = main.get_all_predictions(input_text, top_clean=int(top_k))
        return app.response_class(response=json.dumps(res), status=200, mimetype='application/json')
    except Exception as error:
        err = str(error)
        print(err)
        return app.response_class(response=json.dumps(err), status=500, mimetype='application/json')

if __name__ == '__main__':
    # Start the OSC server in a separate thread
    osc_thread = Thread(target=start_osc_server, daemon=True)
    osc_thread.start()

    # Start the Flask app
    app.run(host='0.0.0.0', debug=True, port=5001, use_reloader=False)

