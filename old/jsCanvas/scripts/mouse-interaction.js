;
(function () {
  if (typeof HyperbolicCanvas === 'undefined') {
    window.HyperbolicCanvas = {};
  }
  if (typeof HyperbolicCanvas.scripts === 'undefined') {
    window.HyperbolicCanvas.scripts = {};
  }
let boolVar = true;
  HyperbolicCanvas.scripts['mouse-interaction'] = function (canvas) {
    var maxN = 12;
    var n = 3;
    var location = HyperbolicCanvas.Point.ORIGIN;
    var rotation = 0;
    var rotationInterval = Math.TAU / 800;
    var radius = 1;

    var ctx = canvas.getContext();
    canvas.setContextProperties({ fillStyle: '#33FFFF' });

    var render = function (event) {
      canvas.clear();

      var polygon = HyperbolicCanvas.Polygon.givenHyperbolicNCenterRadius(n, location, radius, rotation);
      let lines = polygon.getLines();
      let lineLengths = [];
      for (let i = 0; i<lines.length;i++){
        lineLengths.push(lines[i].getHyperbolicLength());
      }


      if (boolVar){
        console.log(polygon)
        boolVar = false;
      }

      if (polygon) {
        var path = canvas.pathForHyperbolic(polygon);
        //var path2 = canvas.pathForHyperbolic(line);

        canvas.fillAndStroke(path);
        
        ctx.fillStyle = 'black'
        ctx.font = "20px Arial";
        ctx.textAlign = 'center';
        let pixelCoord = canvas.getCanvasPixelCoords(lines[0].getHyperbolicMidpoint());
        ctx.fillText(lineLengths[0].toString(), pixelCoord[0],pixelCoord[1]);
        ctx.fillStyle = 'grey'
        //canvas.fillAndStroke(path2);
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
      n += 1;
      n %= maxN;
      n = n < 3 ? 3 : n;
    };

    var scroll = function (event) {
      radius += event.deltaY * .01;
      if (radius < .05) {
        radius = .05;
      } else if (radius > 20) {
        radius = 20;
      }
    };

    canvas.getCanvasElement().addEventListener('click', incrementN);
    canvas.getCanvasElement().addEventListener('mousemove', resetLocation);
    document.addEventListener('wheel', scroll);

    requestAnimationFrame(render);
  };
})();
