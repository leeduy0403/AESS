import { useSocket } from "@/context/SocketContext";
import { useSelector } from "react-redux";
import { UPLOAD_FILE_ROUTE } from "@/utils/constants";
import EmojiPicker from "emoji-picker-react";
import { useEffect, useRef, useState } from "react";
import { GrAttachment } from "react-icons/gr";
import { IoSend } from "react-icons/io5";
import { RiEmojiStickerLine, RiCloseFill } from "react-icons/ri";
import { apiClient } from "@/lib/api-client.js";

function MessageBar({ isReplyingMessage, onCloseReplyMessageClick }) {
  const emojiRef = useRef();
  const fileInputRef = useRef();
  const socket = useSocket()?.current; //TODO figure out why sometimes undefined
  const { selectedChatType, selectedChatData, replyingMessage } = useSelector(
    (state) => state.chat
  );
  const { currentUser } = useSelector((state) => state.user);
  const [message, setMessage] = useState("");
  const [emojiPickerOpen, setEmojiPickerOpen] = useState(false);

  useEffect(() => {
    function handleClickOutside(event) {
      if (emojiRef.current && !emojiRef.current.contains(event.target)) {
        setEmojiPickerOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [emojiRef]);

  const handleAddEmoji = (emoji) => {
    setMessage((msg) => msg + emoji.emoji);
  };
  const handleSendMessage = async () => {
    if (message.trim() === "") return; // Prevent sending empty messages
    if (selectedChatType === "contact") {
      socket.emit("send-message", {
        sender: currentUser._id,
        content: message,
        recipient: selectedChatData._id,
        messageType: "text",
        fileUrl: undefined,
        replyTo: isReplyingMessage ? replyingMessage._id : undefined,
      });
    }
    if (selectedChatType === "channel") {
      console.log(isReplyingMessage ? replyingMessage._id : "NOT REPLYING");
      socket.emit("send-channel-message", {
        sender: currentUser._id,
        content: message,
        messageType: "text",
        fileUrl: undefined,
        replyTo: isReplyingMessage ? replyingMessage._id : undefined,
        channelId: selectedChatData._id,
      });
    }
    onCloseReplyMessageClick(); // Close the reply message box after sending
    setMessage("");
  };

  const handleAttachmentClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleAttachmentChange = async (event) => {
    try {
      const file = event.target.files[0];
      if (file) {
        const formData = new FormData();
        formData.append("file", file);
        const response = await apiClient.post(UPLOAD_FILE_ROUTE, formData, {
          withCredentials: true,
        });

        if (response.status === 200 && response.data) {
          if (selectedChatType === "contact") {
            socket.emit("send-message", {
              sender: currentUser._id,
              content: undefined,
              recipient: selectedChatData._id,
              messageType: "file",
              fileUrl: response.data.filePath,
              replyTo: isReplyingMessage ? replyingMessage._id : undefined,
            });
          }
        }
        if (selectedChatType === "channel") {
          socket.emit("send-channel-message", {
            sender: currentUser._id,
            content: undefined,
            messageType: "file",
            fileUrl: response.data.filePath,
            replyTo: isReplyingMessage ? replyingMessage._id : undefined,
            channelId: selectedChatData._id,
          });
        }
      }
    } catch (error) {
      console.log({ error });
    }
  };

  return (
    <div className="flex flex-col justify-center gap-2">
      {isReplyingMessage && (
        <div className="relative flex items-center bg-[#E8E8E8] px-4 py-2 rounded-md mt-2 mx-8 transition-all duration-100">
          <div className="flex flex-col items-start justify-between">
            <span className="font-semibold text-[#26597C]">
              Replying to{" "}
              {selectedChatType === "contact"
                ? currentUser._id !== replyingMessage.sender
                  ? selectedChatData.name
                  : currentUser.name
                : replyingMessage.sender.name}
            </span>
            <span>
              {replyingMessage.messageType === "text"
                ? replyingMessage.content
                : "file or image"}
            </span>
          </div>
          <div
            className="absolute top-2 right-2 cursor-pointer hover:bg-[#cacaca] rounded-full p-1 transition-all duration-100 mr-1"
            onClick={onCloseReplyMessageClick}
          >
            <RiCloseFill className="text-xl" />
          </div>
        </div>
      )}
      <div
        className={`${
          isReplyingMessage ? "" : "mt-5"
        } overflow-hidden rounded-md mb-6 bg-gradient-to-t flex h-50px items-center justify-center gap-1 bg-white border-2 border-black mx-8`}
      >
        <div className="flex flex-1 items-center gap-5 rounded-md bg-white pr-0">
          <input
            type="text"
            className="flex-1 rounded-md bg-transparent p-4 text-md border-none outline-none focus:border-transparent focus:ring-0"
            placeholder="Aa"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                handleSendMessage();
              }
            }}
          />
          <button
            className="text-neutral-500 transition-all duration-300 hover:text-black focus:border-none focus:text-black focus:outline-none"
            onClick={() => handleAttachmentClick()}
          >
            <GrAttachment className="text-2xl" />
          </button>
          <input
            type="file"
            className="hidden"
            ref={fileInputRef}
            onChange={(e) => handleAttachmentChange(e)}
          />
          <button
            className="text-neutral-500 transition-all duration-300 hover:text-black focus:border-none focus:text-black focus:outline-none items-center justify-center"
            onClick={() =>
              setEmojiPickerOpen((emojiPickerOpen) => !emojiPickerOpen)
            }
          >
            <RiEmojiStickerLine className="text-2xl" />
          </button>
          <div className="">
            <div className="absolute bottom-16 right-0" ref={emojiRef}>
              <EmojiPicker
                theme="light"
                open={emojiPickerOpen}
                onEmojiClick={(emoji) => handleAddEmoji(emoji)}
                autoFocusSearch={false}
              />
            </div>
          </div>
        </div>
        <div className="border-r-2 border-neutral-500 h-[40px]"></div>
        <button
          className="items-center justify-center rounded-md bg-transparent p-4 transition-all duration-300  focus:border-none focus:text-white focus:outline-none"
          onClick={() => handleSendMessage()}
        >
          <IoSend className="text-2xl text-[#26597C] hover:text-[#000000] transition-all duration-100" />
        </button>
      </div>
    </div>
  );
}

export default MessageBar;
