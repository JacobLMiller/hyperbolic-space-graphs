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
  constructor(x,y,id){
    this.x = x*.005;
    this.y = y*.005;
    this.id = id;
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

  HyperbolicCanvas.scripts['text-display'] = function (canvas) {
    //var location = HyperbolicCanvas.Point.ORIGIN;

    let nodes = [];

  //  fr.readAsText(this.files[0[]]);
  console.log('hello');

  nodes.push(new Node(595.05,593.94,"rose"));
  nodes.push(new Node(668.5,638,"dark pink"));
  nodes.push(new Node(400.73,594.14,"peach"));
  nodes.push(new Node(577.97,547.32,"light brown"));
  nodes.push(new Node(556.6,655.8,"salmon"));
  nodes.push(new Node(573.88,503.28,"mauve"));
  nodes.push(new Node(403.86,638.18,"tan"));
  nodes.push(new Node(442.35,463.64,"light purple"));
  nodes.push(new Node(473.9,537.71,"grey"));
  nodes.push(new Node(786.53,666.26,"magenta"));
  nodes.push(new Node(555.37,719.42,"orange"));
  nodes.push(new Node(477.64,582.2,"pink"));
  nodes.push(new Node(765.3,558.14,"purple"));
  nodes.push(new Node(649.92,503.24,"dirt"));
  nodes.push(new Node(742.3,766.92,"hot pink"));
  nodes.push(new Node(181.16,547.71,"pale green"));
  nodes.push(new Node(302.29,464.45,"lilac"));
  nodes.push(new Node(318.59,597.97,"beige"));
  nodes.push(new Node(384.61,508.54,"lavender"));
  nodes.push(new Node(626.11,401.17,"olive"));
  nodes.push(new Node(194.56,597.28,"light green"));
  nodes.push(new Node(213.63,441.66,"light blue"));
  nodes.push(new Node(256.5,397.62,"sky blue"));
  nodes.push(new Node(906.28,819.95,"red"));
  nodes.push(new Node(299.71,553.28,"light pink"));
  nodes.push(new Node(361.29,724.86,"mustard"));
  nodes.push(new Node(264.13,697.2,"lime"));
  nodes.push(new Node(179.23,741.22,"lime green"));
  nodes.push(new Node(242.25,841.3,"yellow"));
  nodes.push(new Node(977.05,507.14,"indigo"));
  nodes.push(new Node(1037.2,448.27,"navy blue"));
  nodes.push(new Node(1160.2,449.06,"royal blue"));
  nodes.push(new Node(883.8,494.88,"maroon"));
  nodes.push(new Node(910.64,450.72,"dark purple"));
  nodes.push(new Node(1045.2,404.25,"dark blue"));
  nodes.push(new Node(782.94,358.12,"forest green"));
  nodes.push(new Node(916.49,320.33,"dark green"));
  nodes.push(new Node(1335.7,423.2,"blue"));
  nodes.push(new Node(706.26,406.42,"brown"));
  nodes.push(new Node(123.2,371.41,"sea green"));
  nodes.push(new Node(129.63,233.74,"aqua"));
  nodes.push(new Node(32.5,256,"cyan"));
  nodes.push(new Node(107.44,189.69,"turquoise"));
  nodes.push(new Node(260.24,149.15,"teal"));
  nodes.push(new Node(384.11,417.42,"periwinkle"));
  nodes.push(new Node(648.17,355.59,"olive green"));
  nodes.push(new Node(500.33,200.49,"green"));
  nodes.push(new Node(1019.4,360.24,"black"));
  nodes.push(new Node(497.43,18,"bright green"));
  nodes.push(new Node(879.76,685.35,"violet"));

    let n = 0;
    let newNodes = []
    let i;
    for(i in nodes){
      newNodes.push(HyperbolicCanvas.Point.givenHyperbolicPolarCoordinates(nodes[i].r,nodes[i].theta));
    }

    let pathList = []

    let edgeList = [["rose","dark pink"],
["rose","peach"],
["rose","light brown"],
["rose","salmon"],
["rose","mauve"],
["rose","tan"],
["rose","light purple"],
["rose","grey"],
["rose","magenta"],
["rose","orange"],
["rose","pink"],
["rose","purple"],
["rose","dirt"],
["dark pink","light brown"],
["dark pink","salmon"],
["dark pink","mauve"],
["dark pink","grey"],
["dark pink","magenta"],
["dark pink","orange"],
["dark pink","pink"],
["dark pink","purple"],
["dark pink","dirt"],
["dark pink","hot pink"],
["peach","light brown"],
["peach","salmon"],
["peach","mauve"],
["peach","tan"],
["peach","grey"],
["peach","pink"],
["peach","pale green"],
["peach","lilac"],
["peach","beige"],
["peach","lavender"],
["peach","light pink"],
["peach","mustard"],
["light brown","salmon"],
["light brown","mauve"],
["light brown","tan"],
["light brown","grey"],
["light brown","orange"],
["light brown","dirt"],
["light brown","olive"],
["light brown","mustard"],
["light brown","olive green"],
["salmon","mauve"],
["salmon","tan"],
["salmon","orange"],
["salmon","pink"],
["mauve","tan"],
["mauve","light purple"],
["mauve","grey"],
["mauve","pink"],
["mauve","purple"],
["mauve","dirt"],
["mauve","lavender"],
["mauve","olive"],
["mauve","periwinkle"],
["tan","grey"],
["tan","pink"],
["tan","dirt"],
["tan","lavender"],
["tan","light green"],
["tan","pale green"],
["tan","beige"],
["tan","light pink"],
["tan","mustard"],
["tan","lime"],
["light purple","grey"],
["light purple","pink"],
["light purple","light blue"],
["light purple","sky blue"],
["light purple","lilac"],
["light purple","lavender"],
["light purple","periwinkle"],
["grey","dirt"],
["grey","light green"],
["grey","light blue"],
["grey","sky blue"],
["grey","lilac"],
["grey","beige"],
["grey","lavender"],
["grey","periwinkle"],
["magenta","purple"],
["magenta","red"],
["magenta","hot pink"],
["magenta","maroon"],
["magenta","violet"],
["orange","mustard"],
["pink","lilac"],
["pink","beige"],
["pink","lavender"],
["pink","light pink"],
["purple","dirt"],
["purple","indigo"],
["purple","maroon"],
["purple","dark purple"],
["purple","violet"],
["dirt","olive"],
["dirt","brown"],
["dirt","olive green"],
["pale green","beige"],
["pale green","light green"],
["pale green","light blue"],
["pale green","light pink"],
["pale green","lime"],
["pale green","sea green"],
["lilac","beige"],
["lilac","lavender"],
["lilac","light blue"],
["lilac","sky blue"],
["lilac","light pink"],
["lilac","periwinkle"],
["beige","lavender"],
["beige","light green"],
["beige","light blue"],
["beige","light pink"],
["lavender","light blue"],
["lavender","sky blue"],
["lavender","light pink"],
["lavender","periwinkle"],
["olive","forest green"],
["olive","brown"],
["olive","green"],
["olive","olive green"],
["light green","light blue"],
["light green","lime"],
["light green","lime green"],
["light green","sea green"],
["light blue","sky blue"],
["light blue","light pink"],
["light blue","sea green"],
["light blue","periwinkle"],
["sky blue","sea green"],
["sky blue","periwinkle"],
["mustard","lime"],
["mustard","lime green"],
["mustard","yellow"],
["lime","lime green"],
["lime","yellow"],
["indigo","navy blue"],
["indigo","royal blue"],
["indigo","maroon"],
["indigo","dark purple"],
["indigo","dark blue"],
["navy blue","royal blue"],
["navy blue","maroon"],
["navy blue","dark purple"],
["navy blue","dark blue"],
["navy blue","forest green"],
["navy blue","dark green"],
["navy blue","black"],
["royal blue","dark purple"],
["royal blue","dark blue"],
["royal blue","blue"],
["maroon","dark purple"],
["maroon","dark blue"],
["maroon","brown"],
["maroon","black"],
["dark purple","dark blue"],
["dark purple","forest green"],
["dark purple","dark green"],
["dark purple","brown"],
["dark purple","black"],
["dark blue","dark green"],
["dark blue","forest green"],
["dark blue","black"],
["forest green","dark green"],
["forest green","brown"],
["forest green","olive green"],
["forest green","black"],
["dark green","brown"],
["dark green","black"],
["brown","olive green"],
["sea green","aqua"],
["sea green","cyan"],
["sea green","turquoise"],
["aqua","cyan"],
["aqua","turquoise"],
["aqua","teal"],
["cyan","turquoise"],
["turquoise","teal"],
["teal","green"],
["olive green","green"],
["green","bright green"]
];

    for (x in edgeList){
      let p0;
      let p1;
      for (i in nodes){
        if (nodes[i].id === edgeList[x][0]){
          p0 = HyperbolicCanvas.Point.givenHyperbolicPolarCoordinates(nodes[i].r,nodes[i].theta);
        }
      }
      for (j in nodes){
        if(nodes[j].id === edgeList[x][1]){
          p1 = HyperbolicCanvas.Point.givenHyperbolicPolarCoordinates(nodes[j].r,nodes[j].theta)
        }
      }
      pathList.push(HyperbolicCanvas.Line.givenTwoPoints(p0,p1));
    };

    var ctx = canvas.getContext();
    console.log(ctx)

    //canvas.setContextProperties(defaultProperties);

    canvas.setContextProperties({ fillStyle: '#66B2FF' });

    var imageData = ctx.getImageData(0,0,1000,1000);

    ctx.font = "30px Comic Sans MS";
    ctx.fillStyle = "red";
    ctx.textAlign = "center";

    var g = graphlibDot.read(
      'graph {\n' +
      '    a -- b;\n' +
      '    }'
    );
    G = document.getElementById("output").innerText

  //  ctx.fillText('Hello world',canvas.width/2,canvas.height);
    console.log(canvas._canvas.height)

    var render = function (event) {
      canvas.clear();

      //ctx.fillText('Hello world',canvas._canvas.height/2,canvas._canvas.height/2);

      //canvas.fill_text('Hellow world',2);


      let i = 0;


      for(i in pathList){
        path = canvas.pathForHyperbolic(pathList[i]);
        canvas.stroke(path);
      }

      i = 0;
      for(i in newNodes){
        path = canvas.pathForHyperbolic(
          HyperbolicCanvas.Circle.givenHyperbolicCenterRadius(newNodes[i],.05)
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
