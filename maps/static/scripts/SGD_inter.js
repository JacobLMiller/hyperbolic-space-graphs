;
(function () {
  if (typeof HyperbolicCanvas === 'undefined') {
    window.HyperbolicCanvas = {};
  }
  if (typeof HyperbolicCanvas.scripts === 'undefined') {
    window.HyperbolicCanvas.scripts = {};
  }

let dragged = false;
let flag = true;
var SCROLL_SPEED = .2;
var SCALE_FACTOR = 1;
let zoom = 1;
let totalZoom = 1;
let myCount = 0;
let bigCount = 0;
let keepGoing = true;
let mapNodes = [];


let sinh = Math.sinh
let cosh = Math.cosh
let sqrt = Math.sqrt
//let graphStr = "graphs/hyperbolic_colors.dot"

let compare = function(a,b){
  if (a.delta < b.delta){
    return -1
  }
  if (a.delta > b.delta){
    return 1
  }
  return 0
}




var newMobius = function(z,transform){
  numerator = math.multiply(z,transform.a)
  numerator = math.add(numerator,transform.b)

  denominator = math.multiply(z,transform.c)
  denominator = math.add(denominator,transform.d)

  fraction = math.divide(numerator,denominator)
  return(fraction)
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
                  //alert(allText);
              }
          }
      }
      rawFile.send(null);
      return(rawFile.response)
  }

var polygonStrToHyperbolic = function(xStr,yStr){
  x = parseFloat(xStr);
  y = parseFloat(yStr);

  //Hardcoded, come back and fix.
  node = new Node(x-originTrans[0],y-originTrans[1]);

  return(lambertAzimuthal(node.r,node.theta));

}


var mobius = function(z,transform){
  z0 = math.Complex.fromPolar(z.getEuclideanRadius(),z.getAngle());
  /*a = transform[0];
  b = transform[1];
  c = transform[2];
  d = transform[3];*/

  if (transform.a != 1){
    numerator = math.multiply(transform.a,z0);
    numerator = math.add(numerator,transform.b);
    denominator = math.multiply(transform.c,z0);
    denominator = math.add(denominator,transform.d);
    newPoint = math.divide(numerator,denominator).toPolar();
    if(newPoint.r > 1){
      return(HyperbolicCanvas.Point.givenEuclideanPolarCoordinates(
        .9999,
        newPoint.phi)
      );
    }
    return(HyperbolicCanvas.Point.givenEuclideanPolarCoordinates(
      newPoint.r,
      newPoint.phi)
    );
  }

  numerator = math.multiply(transform.a,z0);
  numerator = math.add(numerator,transform.b);
  denominator = math.multiply(transform.c,z0);
  denominator = math.add(denominator,transform.d);
  newPoint = math.divide(numerator,denominator).toPolar();
  return(HyperbolicCanvas.Point.givenEuclideanPolarCoordinates(
    newPoint.r,
    newPoint.phi)
  );
}



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
  hR = math.acosh((r*r/2)+1);

  return(HyperbolicCanvas.Point.givenHyperbolicPolarCoordinates(hR,theta));
}


let lobachev_to_h_polar = function(coords){
  let x = coords[0];
  let y = coords[1];

  return {'r': Math.acosh(cosh(x)*cosh(y)),
          't': 2*Math.atanh( sinh(y) / ( sinh(x)*cosh(y) + sqrt( Math.pow(cosh(x),2) * Math.pow(cosh(y), 2) -1) ) )}

}

var makeGraph = function(V,E){
  let i;
  let name;

  let nodeList = [];


  count = 0;
  for (name in V){
    pos = {'x': V[name].pos[0], 'y': V[name].pos[1] }
    console.log(pos)
    //V[name].node = new Node(pos[0]-.5,pos[1]-.5);
    V[name].hPos = HyperbolicCanvas.Point.givenCoordinates(pos.x,pos.y);
    if (V[name].label && V[name].label != "\\N"){
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
    V[name].index = count;
    count += 1;

  nodeList.push(V[name])
  }
  /*myPolygon = generate_N_polygon(nodeList.length);
  for(i in myPolygon){
    nodeList[i].pos = new Node(myPolygon[i][0],myPolygon[i][1]);
    nodeList[i].hPos = lambertAzimuthal(nodeList[i].pos.r,nodeList[i].pos.theta);
  }*/
  //console.log(V)


  let pathList = [];
  for (i in E){
    pathList.push([V[E[i].v],V[E[i].w]]);
  }


  return {
    nodeList,
    pathList
  }

}


  HyperbolicCanvas.scripts['SGD_inter'] = function (canvas,graphStr) {
    var location = HyperbolicCanvas.Point.ORIGIN;
    let n = 0;
    let previousCoverage, currentCoverage, previousZoom, currentZoom;


    let translateX;
    let translateY;

    V = {}
    E = {}

    console.log(graphStr)
    t = JSON.parse(graphStr);
    console.log(t)
    //t = DotParser.parse(readTextFile('graphs/colors.dot'));

    for (i in t.nodes){
      if(true){
        v = t.nodes[i].id
        V[v] = {}
        V[v].neighbors = [];
        V[v].pos = t.nodes[i].pos
        V[v].label = t.nodes[i].label
        if(t.nodes[i].color){
          V[v].color = t.nodes[i].color;
        }
      }
    }
    for(i in t.edges){
      E[i] = {v: t.edges[i].s,
              w: t.edges[i].t}
    }
    for (i in E){
      V[E[i].v].neighbors.push(E[i].w.toString());
      V[E[i].w].neighbors.push(E[i].v.toString());
    }

    let allX = 0;
    let allY = 0;
    let count = 0;
    let totalCount = 0;

    for (name in V){
      //pos = parse_pos(V[name].pos);
      //allX += pos[0];
      //allY += pos[1];
      count += 1;
    }
    console.log(count)

    //originTrans = [allX/count,allY/count];

    let start = performance.now();
    G = makeGraph(V,E);
    //console.log(t);

    // destinationList = [];
    // verticesList = generate_N_polygon(G.nodeList.length);
    // for (i in G.nodeList) {
    //   //position = random_circle_coordinates();
    //   node = new Node(verticesList[i][0],verticesList[i][1]);
    //   //destinationList.push(lambertAzimuthal(node.r,node.theta));
    //   destinationList.push(G.nodeList[i].hPos)
    //   G.nodeList[i].hPos = destinationList[i]
    //   console.log(G.nodeList[i].hPos)
    // }

    //KK_Step(G.nodeList,destinationList);

    var ctx = canvas.getContext();

    //canvas.setContextProperties(defaultProperties);

    var defaultProperties = {
      lineJoin: 'round',
      lineWidth: 1,
      strokeStyle: "#D3D3D3",
      fillStyle: '#66B2FF'
    }

    canvas.setContextProperties(defaultProperties);

    //ctx.fillText('Hello world',100,500);
    var location = HyperbolicCanvas.Point.givenCoordinates(0,0);
    var initialLocation = HyperbolicCanvas.Point.givenCoordinates(0,0);

    var origin = HyperbolicCanvas.Point.givenCoordinates(0,0);
    var render = function (event) {
      canvas.clear();


      //ctx.fillText('Hello world',canvas._canvas.height/2,canvas._canvas.height/2);


      //console.log(canvas.getCanvasPixelCoords(point));

      //Draw edges
      ctx.globalAlpha = parseFloat($("#transparency")[0].value)*.01; //Transparency value
      ctx.lineWidth = 1;
      for(i in G.pathList){
        ctx.strokeStyle = "grey";
        path = canvas.pathForHyperbolic(HyperbolicCanvas.Line.givenTwoPoints(
          G.pathList[i][0].hPos,
          G.pathList[i][1].hPos
        ));
        canvas.stroke(path);
      }
      ctx.globalAlpha = 1;

      //Draw nodes and place text
      nodesAllowed = parseFloat($("#nodesVisible")[0].value)*1;
      for(i in G.nodeList){
        ctx.fillStyle = "grey";
        if(G.nodeList[i].color){
          ctx.fillStyle = G.nodeList[i].color
        }
        path = canvas.pathForHyperbolic(
          HyperbolicCanvas.Circle.givenHyperbolicCenterRadius(G.nodeList[i].hPos,.05)
        );
        canvas.fillAndStroke(path);

        ctx.fillStyle = "black";
        nodeDis = G.nodeList[i].hPos.getEuclideanRadius();

        //Scale text by distance from origin
        if( nodeDis < nodesAllowed){
          fontSize = Math.ceil(nodesAllowed *(1-nodeDis));
          ctx.font = fontSize.toString() + "px Arial";

          let pixelCoord = canvas.getCanvasPixelCoords(G.nodeList[i].hPos);
          ctx.fillText(G.nodeList[i].labelPos.name, pixelCoord[0],pixelCoord[1]);
        }
      }
      count = 0;
      /*for (i in G.nodeList){
        oldPos = G.nodeList[i].hPos;
        newNode = oldPos.hyperbolicDistantPoint(.01,oldPos.hyperbolicAngleTo(destinationList[count]));
        G.nodeList[i].hPos = newNode
        count += 1
      }*/

      totalCount += 1



      requestAnimationFrame(render);
    };


    var resetLocation = function (event) {
      if (event) {
        x = event.clientX;
        y = event.clientY;
      }
      distance = [(translateX-x)*SCROLL_SPEED,(translateY-y)*SCROLL_SPEED];

      location = canvas.getCanvasPixelCoords(location);
      location = canvas.at([location[0] - distance[0], location[1] - distance[1]]);

      changeCenter(location);

    };


var changeCenter = function(center,zoom=1){
  canvas.clear();
  r = center.getEuclideanRadius();
  theta = center.getAngle();

  transform = {
    a: 1,
    b: math.Complex.fromPolar(r,theta).neg(),
    c: math.Complex.fromPolar(r,theta).conjugate().neg(),
    d: 1
  };

  //VERY important to remember
  location = mobius(location,transform);

  for (x in Map.polygonList){
    Map.polygonList[x] = transformPolygon(Map.polygonList[x],transform,zoom);
  }

  for(i in G.nodeList){
    hR = G.nodeList[i].hPos.getHyperbolicRadius()*zoom
    theta = G.nodeList[i].hPos.getAngle()
    G.nodeList[i].hPos = HyperbolicCanvas.Point.givenHyperbolicPolarCoordinates(hR,theta)
    G.nodeList[i].hPos = mobius(G.nodeList[i].hPos,transform);
  }

}



var myDown = function(e){
  dragged = true;
  translateX = e.clientX;
  translateY = e.clientY;
}
var myUp = function(e){
  dragged = false;
  translateX = 0;
  translateY = 0;
}

var whileDragging = function(e){
  if(dragged){
    resetLocation(e);
  }
}

var scroll = function(e){

}

    //canvas.getCanvasElement().addEventListener('click', incrementN);
    canvas.getCanvasElement().addEventListener('mousemove', whileDragging);
    document.addEventListener('wheel', scroll);
    canvas.getCanvasElement().addEventListener('mousedown', myDown);
    canvas.getCanvasElement().addEventListener('mouseup', myUp);

    //console.log(G.nodeList[0])
    //changeCenter(G.nodeList[0].hPos);

    requestAnimationFrame(render);
  };
})();
