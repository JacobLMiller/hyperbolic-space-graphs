;
(function () {
  if (typeof HyperbolicCanvas === 'undefined') {
    window.HyperbolicCanvas = {};
  }
  if (typeof HyperbolicCanvas.scripts === 'undefined') {
    window.HyperbolicCanvas.scripts = {};
  }

let dragged = false;
var SCROLL_SPEED = .01;

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
  node = new Node(x-554.7586,y-496.0374000000002);

  return(lambertAzimuthal(node.r,node.theta));

}

var transformPolygon = function(P,transform){


  vertices = P.getVertices();
  newVertices = [];
  for (i in vertices){
    newVertices.push(mobius(vertices[i],transform));
  }
  /*for (i in vertices){
    p0 = newVertices[i].toPolar();
    p0 = HyperbolicCanvas.Point.givenEuclideanPolarCoordinates(p0.r,p0.phi);
    newVertices[i] = p0;
  }*/
  return(HyperbolicCanvas.Polygon.givenVertices(newVertices))
}

var mobius = function(z,transform){
  z0 = math.Complex.fromPolar(z.getEuclideanRadius(),z.getAngle());
  a = transform[0];
  b = transform[1];
  c = transform[2];
  d = transform[3];

  numerator = math.multiply(a,z0);
  numerator = math.add(numerator,b);
  denominator = math.multiply(c,z0);
  denominator = math.add(denominator,d);
  newPoint = math.divide(numerator,denominator).toPolar();
  return(HyperbolicCanvas.Point.givenEuclideanPolarCoordinates(
    newPoint.r,
    newPoint.phi)
  );
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
  let allX = 0;
  let allY = 0;
  let count = 0;

  for (name in V){
    pos = parse_pos(V[name].pos);
    allX += pos[0];
    allY += pos[1];
    count += 1;
  }

  originTrans = [allX/count,allY/count];

  for (name in V){
    pos = parse_pos(V[name].pos);
    V[name].node = new Node(pos[0]-originTrans[0],pos[1]-originTrans[1]);
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

  HyperbolicCanvas.scripts['scrolling'] = function (canvas) {
    var location = HyperbolicCanvas.Point.ORIGIN;
    let n = 0;

    let translateX;
    let translateY;

    var g = graphlibDot.read(readTextFile("graphs/colors.dot"));
    let V = g._nodes;
    let E = g._edgeObjs;


    t = DotParser.parse(readTextFile("graphs/colors_map.dot"));
    console.log(t.children[0].attr_list[0].eq.trim().split(/\s+/));
  let parsed = true;
   var regions;

   var color;
   var polygonsIdx = 0;
   var colorIdx = 0;
   var lineIdx = 0;
   let colors = [];
   let polygons = [[]];
   let lines = [[]];


   //Parsing code taken from http://gmap.cs.arizona.edu
   regions = t.children[0].attr_list[0].eq.trim().split(/\s+/);
   // parse xdot for region info
   for (var i = 0; i < regions.length; i++) {
       if (regions[i] == "c") { // following specifies color
           i += 2;
           colors[colorIdx] = regions[i];

           if (colors[colorIdx].charAt(0) == '-') { // some color hex's have '-' before
               colors[colorIdx] = colors[colorIdx].substring(1);
           }
           colorIdx++;

       } else if (regions[i] == "P") { // following is a polygon
           i++;
           var size = parseInt(regions[i]); // number of points in polygon

           var polygon = regions.slice(i + 1, i + 1 + size * 2);

           polygon = toCounterClockwise(polygon); // this many dimensions for GeoJson polygon coordinates
           polygons[polygonsIdx++] = polygon;
       } else if (regions[i] == "L") { // following is a line border of the polygon
           i++;
           var size = parseInt(regions[i]);

           var line = regions.slice(i + 1, i + 1 + size * 2);
           lines[lineIdx++] = line;
       }
   }


//console.log(polygons);
myPolygons = [];
for(i=0; i<lines.length; i++){
  myPolygons.push([])
  for (j=0; j < lines[i].length; j+=2){
    myPolygons[i].push(polygonStrToHyperbolic(lines[i][j],lines[i][j+1]));
  }
}

polygonList = [];
for(i = 0; i<myPolygons.length; i++){
  polygonList.push(HyperbolicCanvas.Polygon.givenVertices(myPolygons[i]));
}

    G = makeGraph(V,E);


    var ctx = canvas.getContext();

    //canvas.setContextProperties(defaultProperties);

    canvas.setContextProperties({ fillStyle: '#66B2FF' });

    //ctx.fillText('Hello world',100,500);
    var location = HyperbolicCanvas.Point.givenCoordinates(.1,.1);
    var render = function (event) {
      canvas.clear();



      for(i in polygonList){
        ctx.fillStyle = colors[i];
        path = canvas.pathForHyperbolic(polygonList[i]);
        canvas.fillAndStroke(path);
      }

      //ctx.fillText('Hello world',canvas._canvas.height/2,canvas._canvas.height/2);


      //console.log(canvas.getCanvasPixelCoords(point));

      i = 0;
      /*for(i in G.pathList){
        path = canvas.pathForHyperbolic(G.pathList[i]);
        canvas.stroke(path);
      }
      */

      i = 0;
      for(i in G.nodeList){
        ctx.fillStyle = "grey";
        path = canvas.pathForHyperbolic(
          HyperbolicCanvas.Circle.givenHyperbolicCenterRadius(G.nodeList[i].hPos,.05)
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
      distance = [(translateX-x)*SCROLL_SPEED,(translateY-y)*SCROLL_SPEED];

      location = canvas.getCanvasPixelCoords(location);
      location = canvas.at([location[0] - distance[0], location[1] - distance[1]]);

      changeCenter(location);

    };


var changeCenter = function(center){
  canvas.clear();
  r = center.getEuclideanRadius();
  theta = center.getAngle();

  a = 1;
  b = math.Complex.fromPolar(r,theta).neg();
  c = math.Complex.fromPolar(r,theta).conjugate().neg();
  d = 1;
  transform = [a,b,c,d];

  //VERY important to remember
  location = mobius(location,transform);

  for (x in polygonList){
    polygonList[x] = transformPolygon(polygonList[x],transform);
  }

  //Technically you're doing this n^2 more times than neccessary.
  //Find a way to just redraw the edges at every step.
  for (x in G.pathList){
    p0 = mobius(G.pathList[x].getP0(),transform);
    p1 = mobius(G.pathList[x].getP1(),transform);

    G.pathList[x] = HyperbolicCanvas.Line.givenTwoPoints(p0,p1);
  }

  for(i in G.nodeList){
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


    //canvas.getCanvasElement().addEventListener('click', incrementN);
    canvas.getCanvasElement().addEventListener('mousemove', whileDragging);
    document.addEventListener('wheel', scroll);
    canvas.getCanvasElement().addEventListener('mousedown', myDown);
    canvas.getCanvasElement().addEventListener('mouseup', myUp);




    requestAnimationFrame(render);
  };
})();
