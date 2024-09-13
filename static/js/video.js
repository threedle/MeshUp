dwindow.onload = function() {
    let videos = document.querySelectorAll('video');
    let ended = 0;

    for (let i = 0; i < videos.length; i++) {
        videos[i].addEventListener('ended', function() {
            ended++;
            if (ended === videos.length) {
                for (let j = 0; j < videos.length; j++) {
                    videos[j].currentTime = 0;
                    videos[j].play();
                }
                ended = 0; // reset the counter
            }
        }, false);
    }
}