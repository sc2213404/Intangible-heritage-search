const slider = document.querySelector('.slider');
let autoPlayTimer;


function activate(e) {
    const items = document.querySelectorAll('.item');
    if (e.target.matches('.next')) {
        slider.append(items[0]);
    }
    if (e.target.matches('.prev')) {
        slider.prepend(items[items.length - 1]);
    }
    resetAutoPlay();
}

//手动点击后重置自动播放倒计时
function resetAutoPlay() {
    clearInterval(autoPlayTimer); 
    autoPlayTimer = setInterval(() => {
        const items = document.querySelectorAll('.item');
        slider.append(items[0]);
    }, 4000); 
}


document.addEventListener('click', activate, false);

resetAutoPlay();