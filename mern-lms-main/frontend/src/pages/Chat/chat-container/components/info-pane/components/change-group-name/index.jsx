import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { useState, useEffect } from "react";
import { useSocket } from "@/context/SocketContext.jsx";
import { useSelector } from "react-redux";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button.jsx";
import { FaPen } from "react-icons/fa";

function ChangeGroupName({ currentGroup }) {
  const socket = useSocket().current;
  //   const { selectedChatData } = useSelector((state) => state.chat);	//TODO figure out why not working
  const { currentUser } = useSelector((state) => state.user);
  const [openChangeGroupNameModal, setOpenChangeGroupNameModal] =
    useState(false);
  const [groupName, setGroupName] = useState(currentGroup.name);

  const changeGroupName = (name) => {
    try {
      socket.emit("change-channel-name", {
        sender: currentUser,
        channelId: currentGroup._id,
        name,
      });
    } catch (error) {
      console.log("Error changing group name:", error);
    }
  };

  useEffect(() => {
    setGroupName(currentGroup.name);
  }, [currentGroup]);

  return (
    <>
      <div
        className="py-2 px-2 flex rounded-md gap-5 items-center hover:bg-[#00000022] cursor-pointer"
        onClick={() => setOpenChangeGroupNameModal(true)}
      >
        <div className="bg-[#00000022] rounded-full p-3 flex items-center justify-center">
          <FaPen className="text-black text-xl" />
        </div>
        <span className="text-lg font-medium">Change group name</span>
      </div>
      <Dialog
        open={openChangeGroupNameModal}
        onOpenChange={setOpenChangeGroupNameModal}
      >
        <DialogContent className="flex w-[500px] flex-col border-none bg-[#ffffff]">
          <DialogHeader>
            <DialogTitle className="text-center text-xl">
              Change group name
            </DialogTitle>
            <DialogDescription></DialogDescription>
          </DialogHeader>
          <div>
            <Input
              placeholder="Group name here"
              className="rounded-lg border-2 border-black-20 bg-[#ebebeb] py-6 px-4 transition-all duration-100 focus:bg-white"
              value={groupName}
              onChange={(e) => setGroupName(e.target.value)}
            />
          </div>
          <div className="flex justify-end gap-2">
            <Button
              className="hover:bg-[#E4E4E4] transition-all duration-100 text-lg border"
              onClick={() => {
                setOpenChangeGroupNameModal(false);
                setGroupName(currentGroup.name);
              }}
            >
              Cancel
            </Button>
            <Button
              className="text-white bg-[#26597C] hover:bg-[#1d4762] transition-all duration-100 text-lg"
              onClick={() => {
                changeGroupName(groupName);
                setOpenChangeGroupNameModal(false);
              }}
            >
              Save
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

export default ChangeGroupName;
