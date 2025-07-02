(function() {
  var clientID = 'e1256ea7e23318f';
  
  var initReo = function() {
    Reo.init({
      clientID: clientID
    });
  };
  
  var script = document.createElement('script');
  script.src = 'https://static.reo.dev/' + clientID + '/reo.js';
  script.defer = true;
  script.onload = initReo;
  
  document.head.appendChild(script);
})(); 