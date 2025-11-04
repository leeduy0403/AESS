import { useSelector, useDispatch } from "react-redux";
import { closeChat } from "@/redux/chat/chatSlice";
import { RiCloseFill } from "react-icons/ri";
import { Avatar, AvatarImage } from "@/components/ui/avatar";
import { IoInformationCircleOutline } from "react-icons/io5";

function ChatHeader({ onInfoToggle, showInfo }) {
  const { selectedChatData, selectedChatType } = useSelector(
    (state) => state.chat
  );
  const { currentUser, onlineUsers } = useSelector((state) => state.user);
  const dispatch = useDispatch();

  return (
    <div className="flex h-[10vh] items-center justify-between border-b-2 border-black bg-[#26597C] px-10">
      <div className="flex items-center gap-5">
        <div className="flex items-center justify-center gap-3"></div>
        <div className="flex items-center justify-center gap-5">
          <div className="relative h-10 w-10">
            {selectedChatType === "contact" ? (
              <div className="relative">
                <Avatar className="h-10 w-10 overflow-hidden rounded-full border-black border">
                  <AvatarImage
                    src={selectedChatData.profilePicture}
                    alt="profile"
                    className="h-full w-full bg-white object-cover"
                  />
                </Avatar>
                <div
                  className={`rounded-full w-3 h-3 ml-2 mt-2 absolute bottom-0 right-0 ${
                    onlineUsers?.includes(selectedChatData._id)
                      ? "bg-green-500"
                      : "bg-red-500"
                  }`}
                ></div>
              </div>
            ) : (
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[#E4E4E4] border border-black">
                #
              </div>
            )}
          </div>
          {selectedChatType === "contact" && (
            <span className="text-xl font-semibold text-white">
              {selectedChatData.name
                ? selectedChatData.name
                : selectedChatData.email}
            </span>
          )}
          {selectedChatType === "channel" && (
            <span className="text-lg font-semibold text-white">
              {selectedChatData.name}
            </span>
          )}
          <button className="text-neutral-500 transition-all duration-300 focus:border-none focus:text-white focus:outline-none">
            <RiCloseFill
              className="text-2xl text-white"
              onClick={() => dispatch(closeChat())}
            />
          </button>
        </div>
      </div>
      <div>
        <button
          onClick={onInfoToggle}
          className="transition-all duration-300 focus:outline-none"
          aria-label={
            showInfo ? "Hide chat information" : "Show chat information"
          }
        >
          <IoInformationCircleOutline
            className={`text-2xl ${
              showInfo ? "text-yellow-300" : "text-white"
            }`}
            size={28}
            strokeWidth={1.5}
          />
        </button>
      </div>
    </div>
  );
}

export default ChatHeader;
