document.addEventListener('DOMContentLoaded', function() {
  const bookElement = document.querySelector('.book');

  const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
          if (entry.isIntersecting && entry.intersectionRatio >= 0.5) {
              setTimeout(() => {
                  document.querySelectorAll('.book-item').forEach((item, index) => {
                      // 根据索引选择方向
                      if (index % 2 === 0) {
                          item.classList.add('fade-up'); // 向上移动
                      } else {
                          item.classList.add('fade-out'); // 向下移动
                      }
                  });
              }, 500); // 延时0.5秒
          }
      });
  }, { threshold: 0.5 });
  
  observer.observe(bookElement);
});