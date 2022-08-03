navigator.serviceWorker.register("sw.js");
Notification.requestPermission(result => {
	if (result === "granted") {
		navigator.serviceWorker.ready.then(function(registration) {
			registration.showNotification("Notification with ServiceWorker enabled");
		});
	} else {
		alert("This app requires notification permission to work properly");
	}
});

export function newNotif(title, text) {
	const content = {
		body: text
	};

	navigator.serviceWorker.ready.then(registration => {
		registration.showNotification(title, content);
	});

	setTimeout(() => {
		navigator.serviceWorker.ready.then(registration => {
			registration.getNotifications().then(notifications => {
				for (let notif of notifications) {
					notif?.close()
				}
			})
		});
	}, 9000);
}

export function askNotificationPermission() {
	if (!("Notification" in window)) {
		console.log("This browser does not support notifications.");
	} else {
		Notification.requestPermission();
	}
}