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

  HyperbolicCanvas.scripts['text'] = function (canvas) {
    //var location = HyperbolicCanvas.Point.ORIGIN;

    let nodeList = [];

    nodeList.push(new Node(2.12,.62));
    nodeList.push(new Node(1.2,.55));
    nodeList.push(new Node(1.21,.18));
    nodeList.push(new Node(3.3,1.37));
    nodeList.push(new Node(2.35,1.37));
    nodeList.push(new Node(.275,.621));

    let n = 0;

    newNodes = [];
    let i;
    for(i in nodeList){
      newNodes.push(HyperbolicCanvas.Point.givenHyperbolicPolarCoordinates(nodeList[i].r,nodeList[i].theta));
    }

    let pathList = []

    let edgeList = [[0,1],[1,2],[0,2],[3,4],[5,1],[5,2]];

    for (x in edgeList){
      pathList.push(HyperbolicCanvas.Line.givenTwoPoints(newNodes[edgeList[x][0]],newNodes[edgeList[x][1]]));
    };

    var ctx = canvas.getContext();
    console.log(ctx)

    //canvas.setContextProperties(defaultProperties);

    canvas.setContextProperties({ fillStyle: '#66B2FF' });

    var imageData = ctx.getImageData(0,0,1000,1000);

    ctx.font = "30px Comic Sans MS";
    ctx.fillStyle = "red";
    ctx.textAlign = "center";

    ctx.fillText('Hello world',canvas.width/2,canvas.height);
    console.log(canvas._canvas.height)

    var render = function (event) {
      canvas.clear();

      ctx.fillText('Hello world',canvas._canvas.height/2,canvas._canvas.height/2);

      canvas.fill_text('Hellow world',2);


      let i = 0;


      for(i in pathList){
        path = canvas.pathForHyperbolic(pathList[i]);
        canvas.stroke(path);
      }

      i = 0;
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
      canvas.clear();

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

      for (x in pathList){
        p0 = mobius(pathList[x].getP0(),1,b,c,1);
        p1 = mobius(pathList[x].getP1(),1,b,c,1);

        p0 = p0.toPolar();
        p1 = p1.toPolar();

        p0 = HyperbolicCanvas.Point.givenEuclideanPolarCoordinates(p0.r,p0.phi);
        p1 = HyperbolicCanvas.Point.givenEuclideanPolarCoordinates(p1.r,p1.phi);

        pathList[x] = HyperbolicCanvas.Line.givenTwoPoints(p0,p1);

      }



      n ++;
      if (n >= newNodes.length){
        n = 0
      }
    };



    canvas.getCanvasElement().addEventListener('click', incrementN);
    //canvas.getCanvasElement().addEventListener('mousemove', resetLocation);
    document.addEventListener('wheel', scroll);

    requestAnimationFrame(render);
  };
})();
