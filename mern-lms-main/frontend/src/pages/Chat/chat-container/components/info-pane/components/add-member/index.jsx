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
import { HiUserAdd } from "react-icons/hi";
import { Checkbox } from "@/components/ui/checkbox.jsx";
import { apiClient } from "@/lib/api-client";
import { SEARCH_USER_NOT_IN_CHANNEL_ROUTE } from "@/utils/constants";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarImage } from "@/components/ui/avatar";

function AddMember({ currentGroup }) {
  const socket = useSocket().current;
  const { currentUser } = useSelector((state) => state.user);
  const [openAddMemberModal, setOpenAddMemberModal] = useState(false);
  const [searchedMembers, setSearchedMembers] = useState([]);
  const [selectedMembers, setSelectedMembers] = useState([]);

  const searchUsers = async (searchTerm) => {
    const response = await apiClient.post(
      `${SEARCH_USER_NOT_IN_CHANNEL_ROUTE}/${currentGroup._id}`,
      { searchTerm },
      {
        withCredentials: true,
      }
    );
    if (response.data.users) {
      setSearchedMembers(response.data.users);
    }
  };

  const handleAddMembers = async () => {
    try {
      socket.emit("add-members", {
        sender: currentUser,
        channelId: currentGroup._id,
        memberId: selectedMembers.map((member) => member._id),
        memberName: selectedMembers.map((member) => member.name),
      });
    } catch (error) {
      console.error("Error adding members:", error);
    }
  };

  const closeDialog = () => {
    setOpenAddMemberModal(false);
    setSelectedMembers([]);
    setSearchedMembers([]);
  };

  useEffect(() => {
    if (openAddMemberModal) {
      searchUsers("");
    }
  }, [openAddMemberModal]);

  return (
    <>
      <div
        className="py-2 px-2 flex rounded-md gap-5 items-center hover:bg-[#00000022] cursor-pointer"
        onClick={() => setOpenAddMemberModal(true)}
      >
        <div className="bg-[#00000022] rounded-full p-3 flex items-center justify-center flex-shrink-0">
          <HiUserAdd className="text-black text-xl" />
        </div>
        <span className="text-lg font-medium">Add member</span>
      </div>
      <Dialog open={openAddMemberModal} onOpenChange={setOpenAddMemberModal}>
        <DialogContent className="flex h-[600px] w-[500px] flex-col border-none bg-[#ffffff]">
          <DialogHeader>
            <DialogTitle className="text-center text-xl">
              Add members to "{currentGroup.name}"
            </DialogTitle>
            <DialogDescription></DialogDescription>
          </DialogHeader>
          <div>
            <Input
              placeholder="Search contacts"
              className="rounded-lg border-2 border-black-20 bg-[#ebebeb] p-6 transition-all duration-100 focus:bg-white"
              onChange={(e) => searchUsers(e.target.value)}
            />
          </div>
          {searchedMembers.length > 0 ? (
            <ScrollArea className="h-full">
              <div className="flex flex-col gap-5">
                {searchedMembers.map((member) => (
                  <div
                    key={member._id}
                    className="flex cursor-pointer items-center gap-3"
                    onClick={() =>
                      setSelectedMembers((selectedContacts) => {
                        const isSelected = selectedContacts.some(
                          (selectedContact) =>
                            selectedContact._id === member._id
                        );
                        if (isSelected) {
                          return selectedContacts.filter(
                            (selectedContact) =>
                              selectedContact._id !== member._id
                          );
                        } else {
                          return [...selectedContacts, member];
                        }
                      })
                    }
                  >
                    <Checkbox
                      className="data-[state=checked]:bg-[#26597C] hover:bg-[#1d4762] text-white transition-all duration-100"
                      checked={selectedMembers.some(
                        (selectedContact) => selectedContact._id === member._id
                      )}
                    />
                    <div className="relative h-12 w-12">
                      <Avatar className="h-12 w-12 overflow-hidden rounded-full">
                        <AvatarImage
                          src={member.profilePicture}
                          alt="profile"
                          className="h-full w-full bg-black object-cover"
                        />
                      </Avatar>
                    </div>
                    <div className="flex flex-col">
                      <span>{member.name ? member.name : member.email}</span>
                      <span className="text-xs">{member.email}</span>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          ) : (
            <span className="text-center py-10">No members found</span>
          )}
          <div className="flex justify-end mt-5 gap-2">
            <Button
              className="hover:bg-[#E4E4E4] transition-all duration-100 text-lg border"
              onClick={() => closeDialog()}
            >
              Cancel
            </Button>
            <Button
              className="text-white bg-[#26597C] hover:bg-[#1d4762] transition-all duration-100 text-lg"
              onClick={() => {
                handleAddMembers(selectedMembers);
                closeDialog();
              }}
            >
              Done
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}

export default AddMember;
