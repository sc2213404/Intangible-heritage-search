




window.addEventListener("DOMContentLoaded", function() { 
    const params = new URLSearchParams(window.location.search);
    // console.log("hi")
    const keyword = params.get("keyword");
    console.log("hi")

    console.log("addevent");

    if (keyword) {
        // 更新标题
        document.getElementById("mainTitle").textContent = keyword;
        console.log("h1i")
        // 更新iframe的src
        const descriptionDiv = document.querySelector(".description iframe");
        const encodedKeyword = encodeURIComponent(keyword);
        descriptionDiv.src = `https://baike.baidu.com/item/${encodedKeyword}?fromModule=lemma_search-box`;
        console.log("hi")
        // 获取所有的item2元素并更新其背景图
        const items = document.querySelectorAll(".items2 .item2");
        items.forEach(function(item, index) {
            console.log(index)
            const newImageUrl = `images/images/${keyword}/${index + 1}.jpg`;
            item.style.backgroundImage = `url(${newImageUrl})`;
        });
    }

});



function initMap(keyword) {
    // 创建地图实例
    var map = new BMap.Map("mapContainer");

    var point = new BMap.Point(104.072, 35.70);
    map.centerAndZoom(point, 5);
    
    // 添加地图控件
    map.addControl(new BMap.NavigationControl());    // 添加平移缩放控件
    map.addControl(new BMap.ScaleControl());         // 添加比例尺控件
    map.enableScrollWheelZoom();                     // 启用滚轮放大缩小
    
    // 创建本地搜索实例
    var localSearch = new BMap.LocalSearch(map, {
        renderOptions: {
            map: map,
            autoViewport: true,
            selectFirstResult: true
        },
        pageCapacity: 10,
        onSearchComplete: function(results) {
            if (results && results.getNumPois() > 0) {
                // 可以在这里处理搜索结果
                var firstResult = results.getPoi(0);
                var point = firstResult.point;
                var marker = new BMap.Marker(point);
                map.addOverlay(marker);
                
                var infoWindow = new BMap.InfoWindow(
                    "<div style='padding:10px'>" +
                    "<h4 style='margin:0 0 5px 0'>" + firstResult.title + "</h4>" +
                    "<p style='margin:0;line-height:1.5;font-size:13px'>" +
                    firstResult.address +
                    "</p>" +
                    "</div>"
                );
                
                marker.addEventListener("click", function(){
                    map.openInfoWindow(infoWindow, point);
                });
                
                // 默认打开第一个结果的信息窗口
                map.openInfoWindow(infoWindow, point);
            } else {
                alert('未找到相关地点');
            }
        }
    });
    
    // 如果传入了关键词，立即搜索
    if (keyword) {
        localSearch.search(keyword);
    }
}

// 页面加载完成后初始化地图
window.onload = function() {
    const params = new URLSearchParams(window.location.search);
    const keyword = params.get("keyword");

    initMap(keyword);
};
