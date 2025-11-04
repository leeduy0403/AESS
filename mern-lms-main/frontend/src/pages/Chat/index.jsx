import { useSelector } from "react-redux";
import ContactsContainer from "./contacts-container";
import EmptyChatContainer from "./empty-chat-container";
import ChatContainer from "./chat-container";

function Chat() {
  const { selectedChatData, selectedChatType } = useSelector(
    (state) => state.chat
  );

  return (
    <div className="h-[100vh] w-full bg-[#E4E4E4] border-t border-black flex justify-center pt-20">
      <div className="flex h-[80vh] w-[80vw] gap-1 mx-40 overflow-hidden">
        {/* Show all contact */}
        <ContactsContainer />
        {selectedChatType === undefined ? (
          //   Place holder for not select contact
          <EmptyChatContainer />
        ) : (
          //   Chat content
          <ChatContainer />
        )}
      </div>
    </div>
  );
}
export default Chat;
