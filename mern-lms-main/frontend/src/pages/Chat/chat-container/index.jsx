import ChatHeader from "./components/chat-header";
import MessageBar from "./components/message-bar";
import MessageContainer from "./components/message-container";
import InfoPane from "./components/info-pane/index.jsx";
import { useEffect, useState } from "react";
import { useSelector } from "react-redux";

function ChatContainer() {
  const [showInfo, setShowInfo] = useState(false);
  const [isReplyingMessage, setIsReplyMessage] = useState(false);
  const { selectedChatData } = useSelector((state) => state.chat);

  useEffect(() => {
    setIsReplyMessage(false);
  }, [selectedChatData]);

  return (
    //TODO: Fix responsive UI
    <div className="hidden flex-1 items-center justify-center md:flex rounded-md">
      <div className="flex h-full w-full gap-1 items-start">
        <div
          className={`flex h-full flex-col bg-[#ffffff] flex-1 border-2 border-black rounded-md ${
            showInfo ? "w-[30vw]" : "w-[50vw]"
          }`}
        >
          {/* Passing props for toggle info pane */}
          <ChatHeader
            onInfoToggle={() => setShowInfo(!showInfo)}
            showInfo={showInfo}
          />
          <MessageContainer
            showInfo={showInfo}
            onReplyMessageClick={() => setIsReplyMessage(true)}
          />
          <MessageBar
            isReplyingMessage={isReplyingMessage}
            onCloseReplyMessageClick={() => setIsReplyMessage(false)}
          />
        </div>
        {showInfo && <InfoPane />}
      </div>
    </div>
  );
}

export default ChatContainer;
