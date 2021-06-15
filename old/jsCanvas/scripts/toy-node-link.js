;
(function () {
  if (typeof HyperbolicCanvas === 'undefined') {
    window.HyperbolicCanvas = {};
  }
  if (typeof HyperbolicCanvas.scripts === 'undefined') {
    window.HyperbolicCanvas.scripts = {};
  }
let boolVar = true;

  HyperbolicCanvas.scripts['toy-node-link'] = function (canvas) {
    var location = HyperbolicCanvas.Point.ORIGIN;

    var point1 = HyperbolicCanvas.Point.givenCoordinates(.15,.5);
    var point2 = HyperbolicCanvas.Point.givenCoordinates(-.5,-.5);
    var point3 = HyperbolicCanvas.Point.givenCoordinates(-.5,.5);
    var point4 = HyperbolicCanvas.Point.givenCoordinates(-.3,.25);

    let line = HyperbolicCanvas.Line.givenTwoPoints(point1,point2);
    let line2 = HyperbolicCanvas.Line.givenTwoPoints(point1,point3);
    let line3 = HyperbolicCanvas.Line.givenTwoPoints(point1,point4);
    let line4 = HyperbolicCanvas.Line.givenTwoPoints(point2,point3);



    canvas.setContextProperties({ fillStyle: '#66B2FF' });

    var render = function (event) {
      canvas.clear();

      //var polygon = HyperbolicCanvas.Polygon.givenVertices([point1,point2,point3,point4]);

      if(boolVar){
        console.log(point1);
        boolVar = false;
      }

      path = canvas.pathForHyperbolic(line);
      canvas.stroke(path);

      path = canvas.pathForHyperbolic(line2);
      canvas.stroke(path);

      path = canvas.pathForHyperbolic(line3);
      canvas.stroke(path);

      path = canvas.pathForHyperbolic(line4);
      canvas.stroke(path);

      path = canvas.pathForHyperbolic(
        HyperbolicCanvas.Circle.givenHyperbolicCenterRadius(point1,.1)
      );
      canvas.fillAndStroke(path);

      path = canvas.pathForHyperbolic(
        HyperbolicCanvas.Circle.givenHyperbolicCenterRadius(point2,.1)
      );
      canvas.fillAndStroke(path);


      path = canvas.pathForHyperbolic(
        HyperbolicCanvas.Circle.givenHyperbolicCenterRadius(point3,.1)
      );
      canvas.fillAndStroke(path);


      path = canvas.pathForHyperbolic(
        HyperbolicCanvas.Circle.givenHyperbolicCenterRadius(point4,.1)
      );
      canvas.fillAndStroke(path);

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

    canvas.getCanvasElement().addEventListener('click', incrementN);
    canvas.getCanvasElement().addEventListener('mousemove', resetLocation);
    document.addEventListener('wheel', scroll);

    requestAnimationFrame(render);
  };
})();
