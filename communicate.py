from flask import Flask,request
from flask_cors import cross_origin
from main import main

app = Flask(__name__)
@app.route("/",methods=['GET'])
@cross_origin()
def call_backend():
    mess = main()
    # mess = '0{q0[%f] q1[%f] q2[%f] q3[%f] x[%f] y[%f] z[%f]}'%(q[0],q[1],q[2],q[3],q.x,q.y,q.z)
    print(mess)
    return mess

if __name__ == '__main__':  
    print('hello, world!')
    app.run(debug=True,port=8282) 
