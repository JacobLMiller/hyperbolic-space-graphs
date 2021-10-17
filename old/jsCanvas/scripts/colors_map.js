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
var SCROLL_SPEED = 2;
var SCALE_FACTOR = 1;
let zoom = 1;
let totalZoom = 1;

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

var transformPolygon = function(P,transform,zoom){


  vertices = P.getVertices();
  newVertices = [];
  for (i in vertices){
    hR = vertices[i].getHyperbolicRadius()*zoom
    theta = vertices[i].getAngle()
    vertices[i] = HyperbolicCanvas.Point.givenHyperbolicPolarCoordinates(hR,theta)

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

var parse_pos = function(strPos){
  Coords = strPos.split(',');
  return([parseFloat(Coords[0]),parseFloat(Coords[1])]);
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

var lambertAzimuthal = function(r,theta){
  hR = math.acosh((r*r + 2)/2);

  return(HyperbolicCanvas.Point.givenHyperbolicPolarCoordinates(hR,theta));
}

var makeMap = function(t){
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

if (polygons.length > 0){
  polygons = lines
}
else{
  lines = polygons
}

//console.log(polygons);
myPolygons = [];
for(i=0; i<polygons.length; i++){
 myPolygons.push([])
 for (j=0; j < polygons[i].length; j+=2){

   console.log(polygons[i][j])
   myPolygons[i].push(polygonStrToHyperbolic(polygons[i][j],polygons[i][j+1]));
 }
}

polygonList = [];
for(i = 0; i<myPolygons.length; i++){
 polygonList.push(HyperbolicCanvas.Polygon.givenVertices(myPolygons[i]));
}

console.log(polygonList)

return({
  polygonList: polygonList,
  colors: colors
});

}

var makeGraph = function(V,E){
  let i;
  let name;

  let nodeList = [];



  for (name in V){
    pos = parse_pos(V[name].pos);
    V[name].node = new Node(pos[0],pos[1]);
    V[name].hPos = HyperbolicCanvas.Point.givenHyperbolicPolarCoordinates(V[name].node.r,V[name].node.theta);
    if (V[name].label && V[name].label != "\\N" ){
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
    pathList.push([V[E[i].v],V[E[i].w]]);
  }

  return {
    nodeList,
    pathList
  }

}

  HyperbolicCanvas.scripts['colors_map'] = function (canvas) {
    var location = HyperbolicCanvas.Point.ORIGIN;
    let n = 0;

    let translateX;
    let translateY;

    V = {}
    E = {}
    let aaaa = math.complex(2,3);
    console.log(aaaa.toPolar());
    t = DotParser.parse(readTextFile('graphs/hyperbolic_colors.dot'));

    for (i in t.children){
      //console.log(t.children[i]);
      if(t.children[i].type == 'node_stmt'){
        v = t.children[i].node_id.id
        V[v] = {}
        for(j in t.children[i].attr_list){
          if(t.children[i].attr_list[j].id === 'pos'){
            V[v].pos = t.children[i].attr_list[j].eq
          }
        }
      }
      if(t.children[i].type == 'edge_stmt'){
        E[i] = {v: t.children[i].edge_list[0].id,
                w: t.children[i].edge_list[1].id}
      }
    }

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

    G = makeGraph(V,E);
    //console.log(t);

    /*for(i in t.children){
      if(t.children[i].type === "node_stmt"){
        console.log(t.children[i].node_id.id);
      }
    }*/
    if(t.children[0].attr_list[0].eq){
      Map = makeMap(t);
    }

    console.log(G.nodeList)


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
    var render = function (event) {
      canvas.clear();



      for(i in Map.polygonList){
        ctx.fillStyle = Map.colors[i];
        ctx.strokeStyle = "black";
        path = canvas.pathForHyperbolic(Map.polygonList[i]);
        canvas.fillAndStroke(path);
      }

      //ctx.fillText('Hello world',canvas._canvas.height/2,canvas._canvas.height/2);


      //console.log(canvas.getCanvasPixelCoords(point));

      i = 0;
      for(i in G.pathList){
        ctx.strokeStyle = "grey";
        path = canvas.pathForHyperbolic(HyperbolicCanvas.Line.givenTwoPoints(
          G.pathList[i][0].hPos,
          G.pathList[i][1].hPos
        ));
        canvas.stroke(path);
      }


      i = 0;
      for(i in G.nodeList){
        ctx.fillStyle = "grey";
        path = canvas.pathForHyperbolic(
          HyperbolicCanvas.Circle.givenHyperbolicCenterRadius(G.nodeList[i].hPos,.05)
        );
        canvas.fillAndStroke(path);

        ctx.fillStyle = "black";
        //ctx.font = "20px Arial"
        nodeDis = G.nodeList[i].hPos._euclideanRadius
        if( nodeDis < 1){
          fontSize = Math.ceil(30 *(1-nodeDis));
          ctx.font = fontSize.toString() + "px Arial";

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
  zoom = 1 + e.deltaY*.01;
  totalZoom = totalZoom + e.deltaY*.01;
  if(totalZoom < 300 && totalZoom >-300){
    changeCenter(location,zoom);
  }
  else{
    totalZoom = totalZoom - e.deltaY*.01;
  }
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
