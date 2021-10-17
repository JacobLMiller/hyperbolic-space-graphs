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
//let graphStr = "graphs/hyperbolic_colors.dot"

let L0 = 3;
let K0 = 20;
let ANIMATION_SPEED = 1;
let ITERATIONS = 1000000;

var find_max_delta = function(){
  max_delta = ['dummy',0];
  deltas = 0
  for (m in G.nodeList){
    H = tau(G,m);

    partialDerivative = dEdP(H,m);
    deltas = math.sqrt(partialDerivative[0]*partialDerivative[0] + partialDerivative[1]*partialDerivative[1]);

    if (max_delta[1] < deltas){
      max_delta = [m,deltas];
    }
  }
  return max_delta
}

var generate_N_polygon = function(n){
    verticesList = [];
    theta = (2*3.14)/n
    R = 1
    for (i = 0; i<n; i++){
      verticesList.push([Math.sin(i*theta),math.cos(i*theta)]);
    }
    return verticesList

  }

var dEdP = function(H,p){
  x = 0;
  y = 0;

  for(i in H){
    if (i !== p){
      xDif = H[p][0] - H[i][0];
      yDif = H[p][1] - H[i][1];


      xDif2 = xDif*xDif;
      yDif2 = yDif*yDif;


      denominator = Math.sqrt(xDif2+yDif2);



      k_mi = K[G.nodeList[p].index][G.nodeList[i].index];
      l_mi = L[G.nodeList[p].index][G.nodeList[i].index];

      numerator = l_mi*xDif;
      division = numerator/denominator;

      x += k_mi*(xDif - division);

      numerator = l_mi*yDif;
      division = numerator/denominator;

      y += k_mi*(yDif - division);

    }
  }
  //console.log("dEdP of node " + p.toString())
  //console.log([x,y])
  return [x,y];
}

var dE2 = function(H,p){
  x = 0;
  y = 0;
  xy = 0;
  yx = 0;


  for (i in H){
    //console.log("i is "+i.toString())
    //console.log("p is " + p.toString())
    if (i !== p){
      xDif = H[p][0] - H[i][0];
      yDif = H[p][1] - H[i][1];


      k_mi = K[G.nodeList[p].index][G.nodeList[i].index];
      l_mi = L[G.nodeList[p].index][G.nodeList[i].index];


      denominator = Math.pow(xDif*xDif + yDif*yDif,1.5);

      numerator = l_mi*(yDif*yDif);

      fraction = numerator/denominator
      x += k_mi*(1-fraction);


      numerator = l_mi*xDif*yDif;
      fraction = numerator/denominator;
      xy += k_mi*fraction;


      numerator = l_mi*(xDif*xDif);
      fraction = numerator/denominator;
      y += k_mi*(1-fraction);

    }
  }
  //console.log("dE2 of node " + p.toString())
  return [[x,xy],[xy,y]];
}

var compute_delta_x_and_y = function(H,p){
  //console.log("What is p now " + p.toString())
  coefficientMatrix = dE2(H,p);
  answerMatrix = math.multiply(-1,dEdP(H,p));



  coefficientMatrix = math.matrix(coefficientMatrix);
  answerMatrix = math.matrix(answerMatrix);






  //temp = math.inv(coefficientMatrix);
  return(math.usolve(coefficientMatrix,answerMatrix));
}

var tau = function(G,p){
  z0 = math.complex(G.nodeList[p].hPos.getX(),G.nodeList[p].hPos.getY());

  T = {}
  transform = {
    a: 1,
    b: z0.neg(),
    c: z0.conjugate().neg(),
    d: 1
  }

  for (i in G.nodeList) {
    if(i === p){
      T[i] = [0,0];
      continue
    }

    T[i] = mobius(G.nodeList[i].hPos,transform);


    norm = T[i].getEuclideanRadius();
    angle = T[i].getAngle();

    if (norm >= 1){
      T[i] = HyperbolicCanvas.Point.givenEuclideanPolarCoordinates(.999,T[i].getAngle());
      norm = .999
      continue
    }
    term2 = 2*Math.atanh(norm);

    //temp = math.complex(T[i].getX(),T[i].getY());
    //term1 = math.divide(temp,norm);
    //temp = math.multiply(term1,term2);

    //T[i] = [temp.re,temp.im];
    myComplex = math.Complex.fromPolar(term2,angle);
    T[i] = [myComplex.re,myComplex.im];
  }
  //console.log("Tau at node " + p.toString())
  return T;
}

var newMobius = function(z,transform){
  numerator = math.multiply(z,transform.a)
  numerator = math.add(numerator,transform.b)

  denominator = math.multiply(z,transform.c)
  denominator = math.add(denominator,transform.d)

  fraction = math.divide(numerator,denominator)
  return(fraction)
}

var inverseTau = function(z,z0alt){
  //Probably not right
  norm = z.toPolar().r
  angle = z.toPolar().phi

  //numerator = 1-(Math.pow(Math.E,norm));
  //denominator = 1+(Math.pow(Math.E,norm));

  //division = Math.abs(numerator/denominator);
  division = norm/2;
  division = Math.tanh(division);


  //term1 = math.divide(z,norm);
  //newZ = math.multiply(term1,division);
  //newZ = HyperbolicCanvas.Point.givenCoordinates(newZ.re,newZ.im);
  newZ = math.Complex.fromPolar(division,angle);

  transform = {
    a: -1,
    b: z0alt.neg(),
    c: z0alt.conjugate().neg(),
    d: -1
  }
  answer = newMobius(newZ,transform);

  //answer = mobius(newZ,transform);


  newAnswer = HyperbolicCanvas.Point.givenCoordinates(answer.re,answer.im);

  return(newAnswer);

}

var KK_Step = function(nodes,destinationList,current_max_delta){
  count = 0;
  j = current_max_delta[0]
    //console.log(j)
    z0alt = math.complex(G.nodeList[j].hPos.getX(),G.nodeList[j].hPos.getY());

    H = tau(G,j)

    deltaP = compute_delta_x_and_y(H,j)

    x = deltaP._data[0];
    y = deltaP._data[1];

    z = math.complex(x[0], y[0]);

    //console.log(z0alt)
    destinationList[j] = inverseTau(z,z0alt);

    if(destinationList[j].getEuclideanRadius() >= .999){
      destinationList[j] = HyperbolicCanvas.Point.givenEuclideanPolarCoordinates(.9,destinationList[count].getAngle());
    }
    count += 1
    G.nodeList[j].hPos = destinationList[j];
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

var bfs = function(V,start){
  queue = [start];
  discovered = [start];
  distance = new Map();
  distance.set(start,0);

  while(queue.length > 0){
    v = queue.pop()
    //console.log("We are on node " + V[v].labelPos.name);

    for (w in V[v].neighbors){
      x = V[v].neighbors[w]
      if(discovered.includes(x)){}
      else{
        discovered.push(x);
        //console.log("we have visited " + x.toString());
        distance.set(x,distance.get(v)+1);
        queue.splice(0,0,x);
      }
    }
  }


  myList = [];
  for (node in V){
    if(distance.get(node)){
    myList.push(distance.get(node));
  }else{
    myList.push(L0);
  }

  }
  //console.log(myList);
  return myList;
}

var all_pairs_shortest_path = function(V){
  d = [];

  count = 0;
  for (node in V){
    d.push(bfs(V,node));
    //G.nodeList[node].index = count;
    count += 1;

  }

  return d;
}

var compute_l_table = function(D){
  maximum = 0;
  for(i=0;i<D.length;i++){
    for (j=0;j<D[i].length;j++){
      maximum = Math.max(maximum,D[i][j]);
    }
  }
  L_prime = L0/maximum;

  l = [];
  for (i = 0; i<D.length;i++){
    l.push([]);
    for(j = 0; j<D[i].length; j++){
      l[i].push(L_prime*D[i][j]);
    }
  }
  console.log(l)
  return l;
}

var compute_k = function(D){
  k = [];

  for (i = 0; i<D.length;i++){
    k.push([]);
    for(j = 0; j<D[i].length; j++){
      if(i !== j){
        k[i].push(K0/(D[i][j]*D[i][j]));
      }else{
        k[i].push(0);
      }
    }
  }
  return k;
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


  count = 0;
  for (name in V){
    pos = parse_pos(V[name].pos);
    //V[name].node = new Node(pos[0],pos[1]);
    V[name].hPos = HyperbolicCanvas.Point.givenHyperbolicPolarCoordinates(pos[0],pos[1]);
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

let random_circle_coordinates = function() {
  let x = Math.random();
  let y = Math.random();
  while (x*x + y*y >= 1){
    x = Math.random();
    y = Math.random();
  }
  return([x,y]);
}

  HyperbolicCanvas.scripts['display_output'] = function (canvas) {
    var location = HyperbolicCanvas.Point.ORIGIN;
    let n = 0;
    let previousCoverage, currentCoverage, previousZoom, currentZoom;
    let graphStr = DotParser.parse(readTextFile('graphs/hyperbolic_colors.dot'))


    let translateX;
    let translateY;

    V = {}
    E = {}

    t = graphStr;
    console.log(graphStr)
    //t = DotParser.parse(readTextFile('graphs/colors.dot'));

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
    let totalCount = 0;

    for (name in V){
      //pos = parse_pos(V[name].pos);
      //allX += pos[0];
      //allY += pos[1];
      count += 1;
    }
    console.log(count)

    //originTrans = [allX/count,allY/count];

    G = makeGraph(V,E);
    //console.log(t);

    destinationList = [];
    verticesList = generate_N_polygon(G.nodeList.length);
    for (i in G.nodeList) {
      //position = random_circle_coordinates();
      node = new Node(verticesList[i][0],verticesList[i][1]);
      //destinationList.push(lambertAzimuthal(node.r,node.theta));
      destinationList.push(G.nodeList[i].hPos)
      G.nodeList[i].hPos = destinationList[i]
      console.log(G.nodeList[i].hPos)
    }



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


      for(i in Map.polygonList){
        ctx.fillStyle = Map.colors[i];
        ctx.strokeStyle = "black";
        path = canvas.pathForHyperbolic(Map.polygonList[i]);
        canvas.fillAndStroke(path);
      }

      //ctx.fillText('Hello world',canvas._canvas.height/2,canvas._canvas.height/2);


      //console.log(canvas.getCanvasPixelCoords(point));

      //Draw edges
      //ctx.globalAlpha = parseFloat($("#transparency")[0].value)*.01; //Transparency value
      ctx.lineWidth = 1;
      for(i in G.pathList){
        ctx.strokeStyle = "grey";
        path = canvas.pathForHyperbolic(HyperbolicCanvas.Line.givenTwoPoints(
          G.pathList[i][0].hPos,
          G.pathList[i][1].hPos
        ));
        canvas.stroke(path);
      }
      //ctx.globalAlpha = 1;

      //Draw nodes and place text
      //nodesAllowed = parseFloat($("#nodesVisible")[0].value)*1;
      nodesAllowed = 10;
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
      //path = canvas.pathForHyperbolic(HyperbolicCanvas.Circle.givenHyperbolicCenterRadius(origin,2));
      //canvas.stroke(path);


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
