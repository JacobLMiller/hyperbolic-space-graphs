;
(function () {
  if (typeof HyperbolicCanvas === 'undefined') {
    window.HyperbolicCanvas = {};
  }
  if (typeof HyperbolicCanvas.scripts === 'undefined') {
    window.HyperbolicCanvas.scripts = {};
  }


function toCounterClockwise(polygon) {
    var sum = 0;

    // loop through points and sum edges (x2-x1)(y2+y1)
    for (var i = 0; i + 3 < polygon.length; i += 2) {
        sum += (parseFloat(polygon[i + 2]) - parseFloat(polygon[i])) * (parseFloat(polygon[i + 3]) + parseFloat(polygon[i + 1]));
    }
    // polygon is counterclockwise else convert
    if (sum < 0) {
        return polygon;
    } else {
        // flip array by pairs of points e.g. [x1, y1, x2, y2] -> [x2, y2, x1, y1]
        var result = [];
        for (var i = polygon.length - 2, j = 0; i >= 0; i -= 2, j += 2) {
            result[j] = polygon[i]; // x val
            result[j + 1] = polygon[i + 1]; // y val
        }
    }
    return result
}

function toClockwise(polygon) {
    var sum = 0;

    // loop through points and sum edges (x2-x1)(y2+y1)
    for (var i = 0; i + 3 < polygon.length; i += 2) {
        sum += (parseFloat(polygon[i + 2]) - parseFloat(polygon[i])) * (parseFloat(polygon[i + 3]) + parseFloat(polygon[i + 1]));
    }

    // polygon is counterclockwise else convert
    if (sum >= 0) {
        return polygon; s
    } else {
        // flip array by pairs of points e.g. [x1, y1, x2, y2] -> [x2, y2, x1, y1]
        var result = [];
        for (var i = polygon.length - 2, j = 0; i >= 0; i -= 2, j += 2) {
            result[j] = polygon[i]; // x val
            result[j + 1] = polygon[i + 1]; // y val
        }
    }
    return result
}

function readTextFile(file)
  {
      var rawFile = new XMLHttpRequest();
      rawFile.open("GET", file, false);
      rawFile.onreadystatechange = function ()
      {
          if(rawFile.readyState === 4)
          {
              if(rawFile.status === 200 || rawFile.status == 0)
              {
                  var allText = rawFile.responseText;
                  alert(allText);
              }
          }
      }
      rawFile.send(null);
      return(rawFile.response)
  }

var polygonStrToHyperbolic = function(xStr,yStr){
  x = parseFloat(xStr);
  y = parseFloat(yStr);

  node = new Node(x,y);

  return(lambertAzimuthal(node.r,node.theta));

}


var mobius = function(z,a,b,c,d){
  z0 = math.Complex.fromPolar(z.getEuclideanRadius(),z.getAngle());
  numerator = math.multiply(a,z0);
  numerator = math.add(numerator,b);
  denominator = math.multiply(c,z0);
  denominator = math.add(denominator,d);
  return(math.divide(numerator,denominator));
}

var recenter = function(n){
  /*
  n should be a complex number, the point we are recentering on.
  */
  b = n.neg();
  c = n.neg();
  c = c.conjugate();


}

var parse_pos = function(strPos){
  Coords = strPos.split(',');
  return([parseFloat(Coords[0]),parseFloat(Coords[1])]);
}

var SCALE_FACTOR = .005;

class Node {
  constructor(x,y){
    this.x = x*SCALE_FACTOR;
    this.y = y*SCALE_FACTOR;
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

var lambertAzimuthal = function(r,theta){
  hR = math.acosh((r*r + 2)/2);

  return(HyperbolicCanvas.Point.givenHyperbolicPolarCoordinates(hR,theta));
}

var makeGraph = function(V,E){
  let i;
  let name;

  let nodeList = [];
  for (name in V){
    pos = parse_pos(V[name].pos);
    V[name].node = new Node(pos[0],pos[1]);
    V[name].hPos = lambertAzimuthal(V[name].node.r,V[name].node.theta);
    if (V[name].label){
      V[name].labelPos = {
        name: V[name].label,
        nameLoc: new Node(pos[0],pos[1])
      }
    }
    else{
      V[name].labelPos = {
        name: name,
        nameLoc: new Node(pos[0],pos[1])
      }
    }
  nodeList.push(V[name])
  }


  let pathList = [];
  for (i in E){
    pathList.push(HyperbolicCanvas.Line.givenTwoPoints(
      V[E[i].v].hPos,
      V[E[i].w].hPos
    ));
  }

  return {
    nodeList,
    pathList
  }

}

  HyperbolicCanvas.scripts['text-display'] = function (canvas) {
    var location = HyperbolicCanvas.Point.ORIGIN;
    let n = 0;

    var g = graphlibDot.read(readTextFile("graphs/trad_bubble.dot"));
    let V = g._nodes;
    let E = g._edgeObjs;
    console.log(g);


    G = makeGraph(V,E);


    var ctx = canvas.getContext();

    //canvas.setContextProperties(defaultProperties);

    canvas.setContextProperties({ fillStyle: '#66B2FF' });

    //ctx.fillText('Hello world',100,500);
    var point = HyperbolicCanvas.Point.givenCoordinates(.1,.1);


    var render = function (event) {
      canvas.clear();

      //ctx.fillText('Hello world',canvas._canvas.height/2,canvas._canvas.height/2);

      path = canvas.pathForEuclidean(point);
      canvas.stroke(path);
      //console.log(canvas.getCanvasPixelCoords(point));

      i = 0;
      for(i in G.pathList){
        path = canvas.pathForHyperbolic(G.pathList[i]);
        canvas.stroke(path);
      }

      i = 0;
      for(i in G.nodeList){
        ctx.fillStyle = "#66B2FF";
        path = canvas.pathForHyperbolic(
          HyperbolicCanvas.Circle.givenHyperbolicCenterRadius(G.nodeList[i].hPos,.1)
        );
        canvas.fillAndStroke(path);

        ctx.fillStyle = "black";
        ctx.font = "20px Arial"

        if(G.nodeList[i].hPos._euclideanRadius < .8){
          let pixelCoord = canvas.getCanvasPixelCoords(G.nodeList[i].hPos);
          ctx.fillText(G.nodeList[i].labelPos.name, pixelCoord[0],pixelCoord[1]);
        }
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

      b = math.Complex.fromPolar(G.nodeList[n].hPos.getEuclideanRadius(),G.nodeList[n].hPos.getAngle());
      c = b.clone();
      b = b.neg();
      c = c.conjugate();
      c = c.neg();

      for (x in G.pathList){
        p0 = mobius(G.pathList[x].getP0(),1,b,c,1);
        p1 = mobius(G.pathList[x].getP1(),1,b,c,1);

        p0 = p0.toPolar();
        p1 = p1.toPolar();

        p0 = HyperbolicCanvas.Point.givenEuclideanPolarCoordinates(p0.r,p0.phi);
        p1 = HyperbolicCanvas.Point.givenEuclideanPolarCoordinates(p1.r,p1.phi);

        G.pathList[x] = HyperbolicCanvas.Line.givenTwoPoints(p0,p1);

      }

      for(i in G.nodeList){
        G.nodeList[i].hPos = mobius(G.nodeList[i].hPos,1,b,c,1);
        G.nodeList[i].hPos = G.nodeList[i].hPos.toPolar();
        G.nodeList[i].hPos = HyperbolicCanvas.Point.givenEuclideanPolarCoordinates(G.nodeList[i].hPos.r,G.nodeList[i].hPos.phi);
      }

      n ++;
      if (n >= G.nodeList.length){
        n = 0
      }
    };



    canvas.getCanvasElement().addEventListener('click', incrementN);
    canvas.getCanvasElement().addEventListener('mousemove', resetLocation);
    document.addEventListener('wheel', scroll);

    requestAnimationFrame(render);
  };
})();
