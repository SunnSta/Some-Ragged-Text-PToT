var audio0 = document.getElementById("msc0");
var audio1 = document.getElementById("msc1");
var audio2 = document.getElementById("msc2");
var audio3 = document.getElementById("msc3");
var audio4 = document.getElementById("msc4");

$("#name0").click(function () {
    console.log("play:0");
    audio0.play();
    audio1.pause();
    audio2.pause();
    audio3.pause();
    audio4.pause();
}
);
$("#name1").click(function () {
    console.log("play:1");
    audio1.play();
    audio0.pause();
    audio2.pause();
    audio3.pause();
    audio4.pause();
}
);
$("#name2").click(function () {
    console.log("play:2");
    audio2.play();
    audio1.pause();
    audio0.pause();
    audio3.pause();
    audio4.pause();
}
);
$("#name3").click(function () {
    console.log("play:3");
    audio3.play();
    audio1.pause();
    audio2.pause();
    audio0.pause();
    audio4.pause();
}
);
$("#name4").click(function () {
    console.log("play:4");
    audio4.play();
    audio1.pause();
    audio2.pause();
    audio3.pause();
    audio0.pause();
}
);
