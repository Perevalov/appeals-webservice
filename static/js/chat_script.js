(function () {
    var Message;
    Message = function (arg) {
        this.text = arg.text, this.message_side = arg.message_side;
        this.draw = function (_this) {
            return function () {
                var $message;
                $message = $($('.message_template').clone().html());
                $message.addClass(_this.message_side).find('.text').html(_this.text);
                $('.messages').append($message);
                return setTimeout(function () {
                    return $message.addClass('appeared');
                }, 0);
            };
        }(this);
        return this;
    };
    $(function () {
        var getMessageText, sendMessage,sendRequest,getCookie,setCookie;
        getMessageText = function () {
            var $message_input;
            $message_input = $('.message_input');
            return $message_input.val();
        };
        sendMessage = function (text,message_side) {
            var $messages, message;
            if (text.trim() === '') {
                return;
            }
            $('.message_input').val('');
            $messages = $('.messages');
            message = new Message({
                text: text,
                message_side: message_side
            });
            message.draw();
            return $messages.animate({ scrollTop: $messages.prop('scrollHeight') }, 300);
        };
     
        sendRequest = function (text) {
		$('html, body').animate({
		    scrollTop: $("#sendBtn").offset().top
		  }, 2000);
		  var data = [text];

            $.get("https://appeals-webservice.herokuapp.com/classify", {prediction_data: JSON.stringify(data), user_mode: JSON.stringify(true)}).done(function(data) {
                resp = JSON.parse(data);
                sendMessage(resp['response_text'],'left');
            });
        };

        $('.send_message').click(function (e) {
            var rawText = getMessageText();
            sendMessage(rawText,'right');
            sendRequest(rawText);
        });

        $('.message_input').keyup(function (e) {
            if (e.which === 13) {
                var rawText = getMessageText();
                sendMessage(rawText,'right');
                sendRequest(rawText);
            }
        });
        sendMessage('Привет!','left');
    });
}.call(this));
