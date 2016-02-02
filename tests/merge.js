var rft = $('#rft');
var rtt = $('#rtt');
var cft = $('#cft');
var ctt = $('#ctt');

var gog = [];

document.getElementById('files').addEventListener('change', handleFileSelect, false);

function handleFileSelect(evt) {
gog = [];
files = evt.target.files; // FileList object
// files is a FileList of File objects. List some properties.
var output = [];
for (var i = 0, f; f = files[i]; i++) {
    output.push('<li><strong>', escape(f.name), '</strong> (', f.type || 'n/a', ') - ',
                f.size, ' bytes, last modified: ',
                f.lastModifiedDate ? f.lastModifiedDate.toLocaleDateString() : 'n/a',
                '</li>');
        Load(f);
  }
  document.getElementById('list').innerHTML = '<ul>' + output.join('') + '</ul>';
}


function Load(f) {
var reader = new FileReader();
  // Closure to capture the file information.
  reader.onload = (function(theFile) {
    return function(e) {
      // Render thumbnail.
      console.log(theFile);
      console.log(e);
      var ff = e.target.result.split("\n").slice(rft.val(), rtt.val()+1);
      for (var i = 0; i < ff.length; i++) {
          //console.log(ff[i].split(","));
           if(gog.length <= i){
               gog.push("");
           }
           gog[i] +=  ""+ff[i].split(",").slice(parseInt(cft.val()), parseInt(ctt.val())+1)+",";
      } 
      ff = [];    
    };
  })(f);
  // Read in the image file as a data URL.
  reader.readAsText(f);
}

function ToCSV(o) {
  var s = "";
  for (var i = 0; i < o.length; i++) {
    s+= o[i];
    s+="\r\n";
  }
  return s;
}

function saveTextAsFile(textToWrite)
{
    var textFileAsBlob = new Blob([textToWrite], {type:'text/plain'});
    var fileNameToSaveAs = "data.csv";
    var downloadLink = document.createElement("a");
    downloadLink.download = fileNameToSaveAs;
    downloadLink.innerHTML = "Download File";
    if (window.webkitURL != null)
    {
        // Chrome allows the link to be clicked
        // without actually adding it to the DOM.
        downloadLink.href = window.webkitURL.createObjectURL(textFileAsBlob);
    }
    else
    {
        // Firefox requires the link to be added to the DOM
        // before it can be clicked.
        downloadLink.href = window.URL.createObjectURL(textFileAsBlob);
        downloadLink.onclick = destroyClickedElement;
        downloadLink.style.display = "none";
        document.body.appendChild(downloadLink);
    }
    downloadLink.click();
}
