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

    canvas.setContextProperties({ fillStyle: '#33FFFF' });

    var render = function (event) {
      canvas.clear();

      var polygon = HyperbolicCanvas.Polygon.givenHyperbolicNCenterRadius(n, location, radius, rotation);


      if (boolVar){
        console.log(polygon)
        boolVar = false;
      }

      var point1 = HyperbolicCanvas.Point.givenCoordinates(.5,.5)
      var point2 = HyperbolicCanvas.Point.givenCoordinates(-.5,-.5)

      let line = HyperbolicCanvas.Line.givenTwoPoints(point1,point2);

      if (polygon) {
        var path = canvas.pathForHyperbolic(polygon);
        //var path2 = canvas.pathForHyperbolic(line);
        canvas.fillAndStroke(path);
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
