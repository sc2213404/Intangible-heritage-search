window.addEventListener("load", () => {
	// 先获取一下上传的这个图标元素
	let uploadFile = document.querySelector("#file");
	// 在获取一下整个list和file这个框框
	let list = document.querySelector(".list");
	let file = document.querySelector(".file");

	uploadFile.addEventListener("change", function () {
		for (let item of this.files) {
			// 利用正则表达式判断是否是图片类型;
			if (!/image\/\w+/.test(item.type)) {
				alert("只能选择图片");
				return;
			}

			let reader = new FileReader();
			reader.readAsDataURL(item);
			reader.addEventListener("load", function () {

				let li = document.createElement("li");
				li.innerHTML = `
				<div class="close">×</div>
				<img src="
					${this.result}
					" />
				`;
				list.insertBefore(li, file);
			});
		}
	});

	list.addEventListener("click", (e) => {
		// e是页面一开启就自动生成的事件对象

		// 这里就是如果点击的是×号就删除 否则不删除
		if (e.target.nodeName == "DIV") {
			// removeChild是移除子元素 括号写的是点击的div对应的父元素就是某一个li了
			list.removeChild(e.target.parentNode);
		}
	});

});