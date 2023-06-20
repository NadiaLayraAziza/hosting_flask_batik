function readURL(input) {
    if (input.files && input.files[0]) {
        var filerdr = new FileReader();

        filerdr.onload = function(e) {
            var img = new Image();

            img.onload = function() {
                var canvas = document.createElement('canvas');
                var ctx = canvas.getContext('2d');
                canvas.width = 128;
                canvas.height = 128;
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                // SEND THIS DATA TO WHEREVER YOU NEED IT
                var data = canvas.toDataURL('image/png');

                $('#imageResult').attr('src', img.src);
                //$('#imgprvw').attr('src', data);//converted image in variable 'data'
            }
            img.src = e.target.result;
        }
        filerdr.readAsDataURL(input.files[0]);
    }
}

$(function () {
    $('#upload').on('change', function () {
        readURL(input);
    });
});

/*  ==========================================
    SHOW UPLOADED IMAGE NAME
* ========================================== */
var input = document.getElementById( 'upload' );
var infoArea = document.getElementById( 'upload-label' );

input.addEventListener( 'change', showFileName );
function showFileName( event ) {
  var input = event.srcElement;
  var fileName = input.files[0].name;
  infoArea.textContent = 'File name: ' + fileName;
}