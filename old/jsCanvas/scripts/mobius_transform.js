;
(function () {
  if (typeof HyperbolicCanvas === 'undefined') {
    window.HyperbolicCanvas = {};
  }
  if (typeof HyperbolicCanvas.scripts === 'undefined') {
    window.HyperbolicCanvas.scripts = {};
  }

var mobius = function(z,a,b,c,d){
  z0 = math.Complex.fromPolar(z.getEuclideanRadius(),z.getAngle());
  numerator = math.multiply(a,z0);
  numerator = math.add(numerator,b);
  denominator = math.multiply(c,z0);
  denominator = math.add(denominator,d);
  return(math.divide(numerator,denominator));
}

class Node {
  constructor(x,y){
    this.x = x;
    this.y = y;
    this.r = this.calcR();
    console.log(this.r)
    this.theta = this.calcTheta();
  }

  calcR(){
    return (Math.sqrt(this.x*this.x + this.y*this.y))
  }

  calcTheta(){
    if (this.r === 0){
      console.log("undefined radius");
      return NaN
    }
    if(this.y >= 0){
      return Math.acos(this.x/this.r);
    }
    else {
      return -1*Math.acos(this.x/this.r);
    }
  }

  getR(){
    return (this.r)
  }

}

  HyperbolicCanvas.scripts['mobius_transform'] = function (canvas) {
    var location = HyperbolicCanvas.Point.ORIGIN;

    let x = math.complex(2,4);

    let n = 0;

    newNodes = [];
    newNodes.push(HyperbolicCanvas.Point.givenCoordinates(.5,.1));
    newNodes.push(HyperbolicCanvas.Point.givenCoordinates(.1,-.1));
    newNodes.push(HyperbolicCanvas.Point.givenCoordinates(.3,.1));

    edgeList = [[newNodes[0],newNodes[1]],[newNodes[1],newNodes[2]]];

    //let edgeList = [[node0,node1],[node1,node2],[node0,node2],[node2,node3],[node3,node4],[node5,node1],[node5,node2]];

    canvas.setContextProperties({ fillStyle: '#555555' });

    var render = function (event) {
      canvas.clear();

      let i = 0;
      for(i in newNodes){
        path = canvas.pathForHyperbolic(
          HyperbolicCanvas.Circle.givenHyperbolicCenterRadius(newNodes[i],.1)
        );
        canvas.fillAndStroke(path);
      }


      requestAnimationFrame(render);
    };

    var resetLocation = function (event) {
      if (event) {
        x = event.clientX;
        y = event.clientY;
      }
      location = canvas.at([x, y]);
    };

    var incrementN = function () {
      b = math.Complex.fromPolar(newNodes[n].getEuclideanRadius(),newNodes[n].getAngle());
      c = b.clone();
      b = b.neg();
      c = c.conjugate();
      c = c.neg();
      for(i in newNodes){
        newNodes[i] = mobius(newNodes[i],1,b,c,1);
        newNodes[i] = newNodes[i].toPolar();
        console.log(newNodes[i])
        newNodes[i] = HyperbolicCanvas.Point.givenEuclideanPolarCoordinates(newNodes[i].r,newNodes[i].phi);
      }
      n ++;
    };

    canvas.getCanvasElement().addEventListener('click', incrementN);
    canvas.getCanvasElement().addEventListener('mousemove', resetLocation);
    document.addEventListener('wheel', scroll);

    requestAnimationFrame(render);
  };
})();
